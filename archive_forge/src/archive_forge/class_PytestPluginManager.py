import argparse
import collections.abc
import copy
import dataclasses
import enum
from functools import lru_cache
import glob
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import shlex
import sys
from textwrap import dedent
import types
from types import FunctionType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from pluggy import HookimplMarker
from pluggy import HookimplOpts
from pluggy import HookspecMarker
from pluggy import HookspecOpts
from pluggy import PluginManager
from .compat import PathAwareHookProxy
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup
import _pytest._code
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest._io import TerminalWriter
import _pytest.deprecated
import _pytest.hookspec
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.stash import Stash
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import warn_explicit_for
@final
class PytestPluginManager(PluginManager):
    """A :py:class:`pluggy.PluginManager <pluggy.PluginManager>` with
    additional pytest-specific functionality:

    * Loading plugins from the command line, ``PYTEST_PLUGINS`` env variable and
      ``pytest_plugins`` global variables found in plugins being loaded.
    * ``conftest.py`` loading during start-up.
    """

    def __init__(self) -> None:
        import _pytest.assertion
        super().__init__('pytest')
        self._conftest_plugins: Set[types.ModuleType] = set()
        self._dirpath2confmods: Dict[Path, List[types.ModuleType]] = {}
        self._confcutdir: Optional[Path] = None
        self._noconftest = False
        self._get_directory = lru_cache(256)(_get_directory)
        self.skipped_plugins: List[Tuple[str, str]] = []
        self.add_hookspecs(_pytest.hookspec)
        self.register(self)
        if os.environ.get('PYTEST_DEBUG'):
            err: IO[str] = sys.stderr
            encoding: str = getattr(err, 'encoding', 'utf8')
            try:
                err = open(os.dup(err.fileno()), mode=err.mode, buffering=1, encoding=encoding)
            except Exception:
                pass
            self.trace.root.setwriter(err.write)
            self.enable_tracing()
        self.rewrite_hook = _pytest.assertion.DummyRewriteHook()
        self._configured = False

    def parse_hookimpl_opts(self, plugin: _PluggyPlugin, name: str) -> Optional[HookimplOpts]:
        """:meta private:"""
        if not name.startswith('pytest_'):
            return None
        if name == 'pytest_plugins':
            return None
        opts = super().parse_hookimpl_opts(plugin, name)
        if opts is not None:
            return opts
        method = getattr(plugin, name)
        if not inspect.isroutine(method):
            return None
        return _get_legacy_hook_marks(method, 'impl', ('tryfirst', 'trylast', 'optionalhook', 'hookwrapper'))

    def parse_hookspec_opts(self, module_or_class, name: str) -> Optional[HookspecOpts]:
        """:meta private:"""
        opts = super().parse_hookspec_opts(module_or_class, name)
        if opts is None:
            method = getattr(module_or_class, name)
            if name.startswith('pytest_'):
                opts = _get_legacy_hook_marks(method, 'spec', ('firstresult', 'historic'))
        return opts

    def register(self, plugin: _PluggyPlugin, name: Optional[str]=None) -> Optional[str]:
        if name in _pytest.deprecated.DEPRECATED_EXTERNAL_PLUGINS:
            warnings.warn(PytestConfigWarning('{} plugin has been merged into the core, please remove it from your requirements.'.format(name.replace('_', '-'))))
            return None
        plugin_name = super().register(plugin, name)
        if plugin_name is not None:
            self.hook.pytest_plugin_registered.call_historic(kwargs=dict(plugin=plugin, plugin_name=plugin_name, manager=self))
            if isinstance(plugin, types.ModuleType):
                self.consider_module(plugin)
        return plugin_name

    def getplugin(self, name: str):
        plugin: Optional[_PluggyPlugin] = self.get_plugin(name)
        return plugin

    def hasplugin(self, name: str) -> bool:
        """Return whether a plugin with the given name is registered."""
        return bool(self.get_plugin(name))

    def pytest_configure(self, config: 'Config') -> None:
        """:meta private:"""
        config.addinivalue_line('markers', 'tryfirst: mark a hook implementation function such that the plugin machinery will try to call it first/as early as possible. DEPRECATED, use @pytest.hookimpl(tryfirst=True) instead.')
        config.addinivalue_line('markers', 'trylast: mark a hook implementation function such that the plugin machinery will try to call it last/as late as possible. DEPRECATED, use @pytest.hookimpl(trylast=True) instead.')
        self._configured = True

    def _set_initial_conftests(self, args: Sequence[Union[str, Path]], pyargs: bool, noconftest: bool, rootpath: Path, confcutdir: Optional[Path], invocation_dir: Path, importmode: Union[ImportMode, str], *, consider_namespace_packages: bool) -> None:
        """Load initial conftest files given a preparsed "namespace".

        As conftest files may add their own command line options which have
        arguments ('--my-opt somepath') we might get some false positives.
        All builtin and 3rd party plugins will have been loaded, however, so
        common options will not confuse our logic here.
        """
        self._confcutdir = absolutepath(invocation_dir / confcutdir) if confcutdir else None
        self._noconftest = noconftest
        self._using_pyargs = pyargs
        foundanchor = False
        for intitial_path in args:
            path = str(intitial_path)
            i = path.find('::')
            if i != -1:
                path = path[:i]
            anchor = absolutepath(invocation_dir / path)
            if safe_exists(anchor):
                self._try_load_conftest(anchor, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)
                foundanchor = True
        if not foundanchor:
            self._try_load_conftest(invocation_dir, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)

    def _is_in_confcutdir(self, path: Path) -> bool:
        """Whether to consider the given path to load conftests from."""
        if self._confcutdir is None:
            return True
        return path not in self._confcutdir.parents

    def _try_load_conftest(self, anchor: Path, importmode: Union[str, ImportMode], rootpath: Path, *, consider_namespace_packages: bool) -> None:
        self._loadconftestmodules(anchor, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)
        if anchor.is_dir():
            for x in anchor.glob('test*'):
                if x.is_dir():
                    self._loadconftestmodules(x, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)

    def _loadconftestmodules(self, path: Path, importmode: Union[str, ImportMode], rootpath: Path, *, consider_namespace_packages: bool) -> None:
        if self._noconftest:
            return
        directory = self._get_directory(path)
        if directory in self._dirpath2confmods:
            return
        clist = []
        for parent in reversed((directory, *directory.parents)):
            if self._is_in_confcutdir(parent):
                conftestpath = parent / 'conftest.py'
                if conftestpath.is_file():
                    mod = self._importconftest(conftestpath, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)
                    clist.append(mod)
        self._dirpath2confmods[directory] = clist

    def _getconftestmodules(self, path: Path) -> Sequence[types.ModuleType]:
        directory = self._get_directory(path)
        return self._dirpath2confmods.get(directory, ())

    def _rget_with_confmod(self, name: str, path: Path) -> Tuple[types.ModuleType, Any]:
        modules = self._getconftestmodules(path)
        for mod in reversed(modules):
            try:
                return (mod, getattr(mod, name))
            except AttributeError:
                continue
        raise KeyError(name)

    def _importconftest(self, conftestpath: Path, importmode: Union[str, ImportMode], rootpath: Path, *, consider_namespace_packages: bool) -> types.ModuleType:
        conftestpath_plugin_name = str(conftestpath)
        existing = self.get_plugin(conftestpath_plugin_name)
        if existing is not None:
            return cast(types.ModuleType, existing)
        pkgpath = resolve_package_path(conftestpath)
        if pkgpath is None:
            try:
                del sys.modules[conftestpath.stem]
            except KeyError:
                pass
        try:
            mod = import_path(conftestpath, mode=importmode, root=rootpath, consider_namespace_packages=consider_namespace_packages)
        except Exception as e:
            assert e.__traceback__ is not None
            raise ConftestImportFailure(conftestpath, cause=e) from e
        self._check_non_top_pytest_plugins(mod, conftestpath)
        self._conftest_plugins.add(mod)
        dirpath = conftestpath.parent
        if dirpath in self._dirpath2confmods:
            for path, mods in self._dirpath2confmods.items():
                if dirpath in path.parents or path == dirpath:
                    if mod in mods:
                        raise AssertionError(f'While trying to load conftest path {conftestpath!s}, found that the module {mod} is already loaded with path {mod.__file__}. This is not supposed to happen. Please report this issue to pytest.')
                    mods.append(mod)
        self.trace(f'loading conftestmodule {mod!r}')
        self.consider_conftest(mod, registration_name=conftestpath_plugin_name)
        return mod

    def _check_non_top_pytest_plugins(self, mod: types.ModuleType, conftestpath: Path) -> None:
        if hasattr(mod, 'pytest_plugins') and self._configured and (not self._using_pyargs):
            msg = "Defining 'pytest_plugins' in a non-top-level conftest is no longer supported:\nIt affects the entire test suite instead of just below the conftest as expected.\n  {}\nPlease move it to a top level conftest file at the rootdir:\n  {}\nFor more information, visit:\n  https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files"
            fail(msg.format(conftestpath, self._confcutdir), pytrace=False)

    def consider_preparse(self, args: Sequence[str], *, exclude_only: bool=False) -> None:
        """:meta private:"""
        i = 0
        n = len(args)
        while i < n:
            opt = args[i]
            i += 1
            if isinstance(opt, str):
                if opt == '-p':
                    try:
                        parg = args[i]
                    except IndexError:
                        return
                    i += 1
                elif opt.startswith('-p'):
                    parg = opt[2:]
                else:
                    continue
                parg = parg.strip()
                if exclude_only and (not parg.startswith('no:')):
                    continue
                self.consider_pluginarg(parg)

    def consider_pluginarg(self, arg: str) -> None:
        """:meta private:"""
        if arg.startswith('no:'):
            name = arg[3:]
            if name in essential_plugins:
                raise UsageError('plugin %s cannot be disabled' % name)
            if name == 'cacheprovider':
                self.set_blocked('stepwise')
                self.set_blocked('pytest_stepwise')
            self.set_blocked(name)
            if not name.startswith('pytest_'):
                self.set_blocked('pytest_' + name)
        else:
            name = arg
            self.unblock(name)
            if not name.startswith('pytest_'):
                self.unblock('pytest_' + name)
            self.import_plugin(arg, consider_entry_points=True)

    def consider_conftest(self, conftestmodule: types.ModuleType, registration_name: str) -> None:
        """:meta private:"""
        self.register(conftestmodule, name=registration_name)

    def consider_env(self) -> None:
        """:meta private:"""
        self._import_plugin_specs(os.environ.get('PYTEST_PLUGINS'))

    def consider_module(self, mod: types.ModuleType) -> None:
        """:meta private:"""
        self._import_plugin_specs(getattr(mod, 'pytest_plugins', []))

    def _import_plugin_specs(self, spec: Union[None, types.ModuleType, str, Sequence[str]]) -> None:
        plugins = _get_plugin_specs_as_list(spec)
        for import_spec in plugins:
            self.import_plugin(import_spec)

    def import_plugin(self, modname: str, consider_entry_points: bool=False) -> None:
        """Import a plugin with ``modname``.

        If ``consider_entry_points`` is True, entry point names are also
        considered to find a plugin.
        """
        assert isinstance(modname, str), 'module name as text required, got %r' % modname
        if self.is_blocked(modname) or self.get_plugin(modname) is not None:
            return
        importspec = '_pytest.' + modname if modname in builtin_plugins else modname
        self.rewrite_hook.mark_rewrite(importspec)
        if consider_entry_points:
            loaded = self.load_setuptools_entrypoints('pytest11', name=modname)
            if loaded:
                return
        try:
            __import__(importspec)
        except ImportError as e:
            raise ImportError(f'Error importing plugin "{modname}": {e.args[0]}').with_traceback(e.__traceback__) from e
        except Skipped as e:
            self.skipped_plugins.append((modname, e.msg or ''))
        else:
            mod = sys.modules[importspec]
            self.register(mod, modname)