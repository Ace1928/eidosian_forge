from __future__ import annotations
import importlib.util
import inspect
import itertools
import os
import platform
import re
import sys
import sysconfig
import traceback
from types import FrameType, ModuleType
from typing import (
from coverage import env
from coverage.disposition import FileDisposition, disposition_init
from coverage.exceptions import CoverageException, PluginError
from coverage.files import TreeMatcher, GlobMatcher, ModuleMatcher
from coverage.files import prep_patterns, find_python_files, canonical_filename
from coverage.misc import sys_modules_saved
from coverage.python import source_for_file, source_for_morf
from coverage.types import TFileDisposition, TMorf, TWarnFn, TDebugCtl
class InOrOut:
    """Machinery for determining what files to measure."""

    def __init__(self, config: CoverageConfig, warn: TWarnFn, debug: TDebugCtl | None, include_namespace_packages: bool) -> None:
        self.warn = warn
        self.debug = debug
        self.include_namespace_packages = include_namespace_packages
        self.source: list[str] = []
        self.source_pkgs: list[str] = []
        self.source_pkgs.extend(config.source_pkgs)
        for src in config.source or []:
            if os.path.isdir(src):
                self.source.append(canonical_filename(src))
            else:
                self.source_pkgs.append(src)
        self.source_pkgs_unmatched = self.source_pkgs[:]
        self.include = prep_patterns(config.run_include)
        self.omit = prep_patterns(config.run_omit)
        self.pylib_paths: set[str] = set()
        if not config.cover_pylib:
            add_stdlib_paths(self.pylib_paths)
        self.cover_paths: set[str] = set()
        add_coverage_paths(self.cover_paths)
        self.third_paths: set[str] = set()
        add_third_party_paths(self.third_paths)

        def _debug(msg: str) -> None:
            if self.debug:
                self.debug.write(msg)
        _debug('sys.path:' + ''.join((f'\n    {p}' for p in sys.path)))
        self.source_match = None
        self.source_pkgs_match = None
        self.pylib_match = None
        self.include_match = self.omit_match = None
        if self.source or self.source_pkgs:
            against = []
            if self.source:
                self.source_match = TreeMatcher(self.source, 'source')
                against.append(f'trees {self.source_match!r}')
            if self.source_pkgs:
                self.source_pkgs_match = ModuleMatcher(self.source_pkgs, 'source_pkgs')
                against.append(f'modules {self.source_pkgs_match!r}')
            _debug('Source matching against ' + ' and '.join(against))
        elif self.pylib_paths:
            self.pylib_match = TreeMatcher(self.pylib_paths, 'pylib')
            _debug(f'Python stdlib matching: {self.pylib_match!r}')
        if self.include:
            self.include_match = GlobMatcher(self.include, 'include')
            _debug(f'Include matching: {self.include_match!r}')
        if self.omit:
            self.omit_match = GlobMatcher(self.omit, 'omit')
            _debug(f'Omit matching: {self.omit_match!r}')
        self.cover_match = TreeMatcher(self.cover_paths, 'coverage')
        _debug(f'Coverage code matching: {self.cover_match!r}')
        self.third_match = TreeMatcher(self.third_paths, 'third')
        _debug(f'Third-party lib matching: {self.third_match!r}')
        self.source_in_third_paths = set()
        with sys_modules_saved():
            for pkg in self.source_pkgs:
                try:
                    modfile, path = file_and_path_for_module(pkg)
                    _debug(f'Imported source package {pkg!r} as {modfile!r}')
                except CoverageException as exc:
                    _debug(f"Couldn't import source package {pkg!r}: {exc}")
                    continue
                if modfile:
                    if self.third_match.match(modfile):
                        _debug(f'Source in third-party: source_pkg {pkg!r} at {modfile!r}')
                        self.source_in_third_paths.add(canonical_path(source_for_file(modfile)))
                else:
                    for pathdir in path:
                        if self.third_match.match(pathdir):
                            _debug(f'Source in third-party: {pkg!r} path directory at {pathdir!r}')
                            self.source_in_third_paths.add(pathdir)
        for src in self.source:
            if self.third_match.match(src):
                _debug(f'Source in third-party: source directory {src!r}')
                self.source_in_third_paths.add(src)
        self.source_in_third_match = TreeMatcher(self.source_in_third_paths, 'source_in_third')
        _debug(f'Source in third-party matching: {self.source_in_third_match}')
        self.plugins: Plugins
        self.disp_class: type[TFileDisposition] = FileDisposition

    def should_trace(self, filename: str, frame: FrameType | None=None) -> TFileDisposition:
        """Decide whether to trace execution in `filename`, with a reason.

        This function is called from the trace function.  As each new file name
        is encountered, this function determines whether it is traced or not.

        Returns a FileDisposition object.

        """
        original_filename = filename
        disp = disposition_init(self.disp_class, filename)

        def nope(disp: TFileDisposition, reason: str) -> TFileDisposition:
            """Simple helper to make it easy to return NO."""
            disp.trace = False
            disp.reason = reason
            return disp
        if original_filename.startswith('<'):
            return nope(disp, 'original file name is not real')
        if frame is not None:
            dunder_file = frame.f_globals and frame.f_globals.get('__file__')
            if dunder_file:
                filename = source_for_file(dunder_file)
                if original_filename and (not original_filename.startswith('<')):
                    orig = os.path.basename(original_filename)
                    if orig != os.path.basename(filename):
                        filename = original_filename
        if not filename:
            return nope(disp, "empty string isn't a file name")
        if filename.startswith('memory:'):
            return nope(disp, "memory isn't traceable")
        if filename.startswith('<'):
            return nope(disp, 'file name is not real')
        canonical = canonical_filename(filename)
        disp.canonical_filename = canonical
        plugin = None
        for plugin in self.plugins.file_tracers:
            if not plugin._coverage_enabled:
                continue
            try:
                file_tracer = plugin.file_tracer(canonical)
                if file_tracer is not None:
                    file_tracer._coverage_plugin = plugin
                    disp.trace = True
                    disp.file_tracer = file_tracer
                    if file_tracer.has_dynamic_source_filename():
                        disp.has_dynamic_filename = True
                    else:
                        disp.source_filename = canonical_filename(file_tracer.source_filename())
                    break
            except Exception:
                plugin_name = plugin._coverage_plugin_name
                tb = traceback.format_exc()
                self.warn(f'Disabling plug-in {plugin_name!r} due to an exception:\n{tb}')
                plugin._coverage_enabled = False
                continue
        else:
            disp.trace = True
            disp.source_filename = canonical
        if not disp.has_dynamic_filename:
            if not disp.source_filename:
                raise PluginError(f"Plugin {plugin!r} didn't set source_filename for '{disp.original_filename}'")
            reason = self.check_include_omit_etc(disp.source_filename, frame)
            if reason:
                nope(disp, reason)
        return disp

    def check_include_omit_etc(self, filename: str, frame: FrameType | None) -> str | None:
        """Check a file name against the include, omit, etc, rules.

        Returns a string or None.  String means, don't trace, and is the reason
        why.  None means no reason found to not trace.

        """
        modulename = name_for_module(filename, frame)
        if self.source_match or self.source_pkgs_match:
            extra = ''
            ok = False
            if self.source_pkgs_match:
                if self.source_pkgs_match.match(modulename):
                    ok = True
                    if modulename in self.source_pkgs_unmatched:
                        self.source_pkgs_unmatched.remove(modulename)
                else:
                    extra = f'module {modulename!r} '
            if not ok and self.source_match:
                if self.source_match.match(filename):
                    ok = True
            if not ok:
                return extra + 'falls outside the --source spec'
            if self.third_match.match(filename) and (not self.source_in_third_match.match(filename)):
                return 'inside --source, but is third-party'
        elif self.include_match:
            if not self.include_match.match(filename):
                return 'falls outside the --include trees'
        else:
            if self.cover_match.match(filename):
                return 'is part of coverage.py'
            if self.pylib_match and self.pylib_match.match(filename):
                return 'is in the stdlib'
            if self.third_match.match(filename):
                return 'is a third-party module'
        if self.omit_match and self.omit_match.match(filename):
            return 'is inside an --omit pattern'
        try:
            filename.encode('utf-8')
        except UnicodeEncodeError:
            return 'non-encodable filename'
        return None

    def warn_conflicting_settings(self) -> None:
        """Warn if there are settings that conflict."""
        if self.include:
            if self.source or self.source_pkgs:
                self.warn('--include is ignored because --source is set', slug='include-ignored')

    def warn_already_imported_files(self) -> None:
        """Warn if files have already been imported that we will be measuring."""
        if self.include or self.source or self.source_pkgs:
            warned = set()
            for mod in list(sys.modules.values()):
                filename = getattr(mod, '__file__', None)
                if filename is None:
                    continue
                if filename in warned:
                    continue
                if len(getattr(mod, '__path__', ())) > 1:
                    continue
                disp = self.should_trace(filename)
                if disp.has_dynamic_filename:
                    continue
                if disp.trace:
                    msg = f'Already imported a file that will be measured: {filename}'
                    self.warn(msg, slug='already-imported')
                    warned.add(filename)
                elif self.debug and self.debug.should('trace'):
                    self.debug.write("Didn't trace already imported file {!r}: {}".format(disp.original_filename, disp.reason))

    def warn_unimported_source(self) -> None:
        """Warn about source packages that were of interest, but never traced."""
        for pkg in self.source_pkgs_unmatched:
            self._warn_about_unmeasured_code(pkg)

    def _warn_about_unmeasured_code(self, pkg: str) -> None:
        """Warn about a package or module that we never traced.

        `pkg` is a string, the name of the package or module.

        """
        mod = sys.modules.get(pkg)
        if mod is None:
            self.warn(f'Module {pkg} was never imported.', slug='module-not-imported')
            return
        if module_is_namespace(mod):
            return
        if not module_has_file(mod):
            self.warn(f'Module {pkg} has no Python source.', slug='module-not-python')
            return
        msg = f'Module {pkg} was previously imported, but not measured'
        self.warn(msg, slug='module-not-measured')

    def find_possibly_unexecuted_files(self) -> Iterable[tuple[str, str | None]]:
        """Find files in the areas of interest that might be untraced.

        Yields pairs: file path, and responsible plug-in name.
        """
        for pkg in self.source_pkgs:
            if pkg not in sys.modules or not module_has_file(sys.modules[pkg]):
                continue
            pkg_file = source_for_file(cast(str, sys.modules[pkg].__file__))
            yield from self._find_executable_files(canonical_path(pkg_file))
        for src in self.source:
            yield from self._find_executable_files(src)

    def _find_plugin_files(self, src_dir: str) -> Iterable[tuple[str, str]]:
        """Get executable files from the plugins."""
        for plugin in self.plugins.file_tracers:
            for x_file in plugin.find_executable_files(src_dir):
                yield (x_file, plugin._coverage_plugin_name)

    def _find_executable_files(self, src_dir: str) -> Iterable[tuple[str, str | None]]:
        """Find executable files in `src_dir`.

        Search for files in `src_dir` that can be executed because they
        are probably importable. Don't include ones that have been omitted
        by the configuration.

        Yield the file path, and the plugin name that handles the file.

        """
        py_files = ((py_file, None) for py_file in find_python_files(src_dir, self.include_namespace_packages))
        plugin_files = self._find_plugin_files(src_dir)
        for file_path, plugin_name in itertools.chain(py_files, plugin_files):
            file_path = canonical_filename(file_path)
            if self.omit_match and self.omit_match.match(file_path):
                continue
            yield (file_path, plugin_name)

    def sys_info(self) -> Iterable[tuple[str, Any]]:
        """Our information for Coverage.sys_info.

        Returns a list of (key, value) pairs.
        """
        info = [('coverage_paths', self.cover_paths), ('stdlib_paths', self.pylib_paths), ('third_party_paths', self.third_paths), ('source_in_third_party_paths', self.source_in_third_paths)]
        matcher_names = ['source_match', 'source_pkgs_match', 'include_match', 'omit_match', 'cover_match', 'pylib_match', 'third_match', 'source_in_third_match']
        for matcher_name in matcher_names:
            matcher = getattr(self, matcher_name)
            if matcher:
                matcher_info = matcher.info()
            else:
                matcher_info = '-none-'
            info.append((matcher_name, matcher_info))
        return info