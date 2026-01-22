import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class ConfigDiscovery:
    """Fill-in metadata and options that can be automatically derived
    (from other metadata/options, the file system or conventions)
    """

    def __init__(self, distribution: 'Distribution'):
        self.dist = distribution
        self._called = False
        self._disabled = False
        self._skip_ext_modules = False

    def _disable(self):
        """Internal API to disable automatic discovery"""
        self._disabled = True

    def _ignore_ext_modules(self):
        """Internal API to disregard ext_modules.

        Normally auto-discovery would not be triggered if ``ext_modules`` are set
        (this is done for backward compatibility with existing packages relying on
        ``setup.py`` or ``setup.cfg``). However, ``setuptools`` can call this function
        to ignore given ``ext_modules`` and proceed with the auto-discovery if
        ``packages`` and ``py_modules`` are not given (e.g. when using pyproject.toml
        metadata).
        """
        self._skip_ext_modules = True

    @property
    def _root_dir(self) -> _Path:
        return self.dist.src_root or os.curdir

    @property
    def _package_dir(self) -> Dict[str, str]:
        if self.dist.package_dir is None:
            return {}
        return self.dist.package_dir

    def __call__(self, force=False, name=True, ignore_ext_modules=False):
        """Automatically discover missing configuration fields
        and modifies the given ``distribution`` object in-place.

        Note that by default this will only have an effect the first time the
        ``ConfigDiscovery`` object is called.

        To repeatedly invoke automatic discovery (e.g. when the project
        directory changes), please use ``force=True`` (or create a new
        ``ConfigDiscovery`` instance).
        """
        if force is False and (self._called or self._disabled):
            return
        self._analyse_package_layout(ignore_ext_modules)
        if name:
            self.analyse_name()
        self._called = True

    def _explicitly_specified(self, ignore_ext_modules: bool) -> bool:
        """``True`` if the user has specified some form of package/module listing"""
        ignore_ext_modules = ignore_ext_modules or self._skip_ext_modules
        ext_modules = not (self.dist.ext_modules is None or ignore_ext_modules)
        return self.dist.packages is not None or self.dist.py_modules is not None or ext_modules or (hasattr(self.dist, 'configuration') and self.dist.configuration)

    def _analyse_package_layout(self, ignore_ext_modules: bool) -> bool:
        if self._explicitly_specified(ignore_ext_modules):
            return True
        log.debug('No `packages` or `py_modules` configuration, performing automatic discovery.')
        return self._analyse_explicit_layout() or self._analyse_src_layout() or self._analyse_flat_layout()

    def _analyse_explicit_layout(self) -> bool:
        """The user can explicitly give a package layout via ``package_dir``"""
        package_dir = self._package_dir.copy()
        package_dir.pop('', None)
        root_dir = self._root_dir
        if not package_dir:
            return False
        log.debug(f'`explicit-layout` detected -- analysing {package_dir}')
        pkgs = chain_iter((_find_packages_within(pkg, os.path.join(root_dir, parent_dir)) for pkg, parent_dir in package_dir.items()))
        self.dist.packages = list(pkgs)
        log.debug(f'discovered packages -- {self.dist.packages}')
        return True

    def _analyse_src_layout(self) -> bool:
        """Try to find all packages or modules under the ``src`` directory
        (or anything pointed by ``package_dir[""]``).

        The "src-layout" is relatively safe for automatic discovery.
        We assume that everything within is meant to be included in the
        distribution.

        If ``package_dir[""]`` is not given, but the ``src`` directory exists,
        this function will set ``package_dir[""] = "src"``.
        """
        package_dir = self._package_dir
        src_dir = os.path.join(self._root_dir, package_dir.get('', 'src'))
        if not os.path.isdir(src_dir):
            return False
        log.debug(f'`src-layout` detected -- analysing {src_dir}')
        package_dir.setdefault('', os.path.basename(src_dir))
        self.dist.package_dir = package_dir
        self.dist.packages = PEP420PackageFinder.find(src_dir)
        self.dist.py_modules = ModuleFinder.find(src_dir)
        log.debug(f'discovered packages -- {self.dist.packages}')
        log.debug(f'discovered py_modules -- {self.dist.py_modules}')
        return True

    def _analyse_flat_layout(self) -> bool:
        """Try to find all packages and modules under the project root.

        Since the ``flat-layout`` is more dangerous in terms of accidentally including
        extra files/directories, this function is more conservative and will raise an
        error if multiple packages or modules are found.

        This assumes that multi-package dists are uncommon and refuse to support that
        use case in order to be able to prevent unintended errors.
        """
        log.debug(f'`flat-layout` detected -- analysing {self._root_dir}')
        return self._analyse_flat_packages() or self._analyse_flat_modules()

    def _analyse_flat_packages(self) -> bool:
        self.dist.packages = FlatLayoutPackageFinder.find(self._root_dir)
        top_level = remove_nested_packages(remove_stubs(self.dist.packages))
        log.debug(f'discovered packages -- {self.dist.packages}')
        self._ensure_no_accidental_inclusion(top_level, 'packages')
        return bool(top_level)

    def _analyse_flat_modules(self) -> bool:
        self.dist.py_modules = FlatLayoutModuleFinder.find(self._root_dir)
        log.debug(f'discovered py_modules -- {self.dist.py_modules}')
        self._ensure_no_accidental_inclusion(self.dist.py_modules, 'modules')
        return bool(self.dist.py_modules)

    def _ensure_no_accidental_inclusion(self, detected: List[str], kind: str):
        if len(detected) > 1:
            from inspect import cleandoc
            from setuptools.errors import PackageDiscoveryError
            msg = f'Multiple top-level {kind} discovered in a flat-layout: {detected}.\n\n            To avoid accidental inclusion of unwanted files or directories,\n            setuptools will not proceed with this build.\n\n            If you are trying to create a single distribution with multiple {kind}\n            on purpose, you should not rely on automatic discovery.\n            Instead, consider the following options:\n\n            1. set up custom discovery (`find` directive with `include` or `exclude`)\n            2. use a `src-layout`\n            3. explicitly set `py_modules` or `packages` with a list of names\n\n            To find more information, look for "package discovery" on setuptools docs.\n            '
            raise PackageDiscoveryError(cleandoc(msg))

    def analyse_name(self):
        """The packages/modules are the essential contribution of the author.
        Therefore the name of the distribution can be derived from them.
        """
        if self.dist.metadata.name or self.dist.name:
            return
        log.debug('No `name` configuration, performing automatic discovery')
        name = self._find_name_single_package_or_module() or self._find_name_from_packages()
        if name:
            self.dist.metadata.name = name

    def _find_name_single_package_or_module(self) -> Optional[str]:
        """Exactly one module or package"""
        for field in ('packages', 'py_modules'):
            items = getattr(self.dist, field, None) or []
            if items and len(items) == 1:
                log.debug(f'Single module/package detected, name: {items[0]}')
                return items[0]
        return None

    def _find_name_from_packages(self) -> Optional[str]:
        """Try to find the root package that is not a PEP 420 namespace"""
        if not self.dist.packages:
            return None
        packages = remove_stubs(sorted(self.dist.packages, key=len))
        package_dir = self.dist.package_dir or {}
        parent_pkg = find_parent_package(packages, package_dir, self._root_dir)
        if parent_pkg:
            log.debug(f'Common parent package detected, name: {parent_pkg}')
            return parent_pkg
        log.warn('No parent package detected, impossible to derive `name`')
        return None