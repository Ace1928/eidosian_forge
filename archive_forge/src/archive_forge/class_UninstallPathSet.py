import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
class UninstallPathSet:
    """A set of file paths to be removed in the uninstallation of a
    requirement."""

    def __init__(self, dist: BaseDistribution) -> None:
        self._paths: Set[str] = set()
        self._refuse: Set[str] = set()
        self._pth: Dict[str, UninstallPthEntries] = {}
        self._dist = dist
        self._moved_paths = StashedUninstallPathSet()
        self._normalize_path_cached = functools.lru_cache()(normalize_path)

    def _permitted(self, path: str) -> bool:
        """
        Return True if the given path is one we are permitted to
        remove/modify, False otherwise.

        """
        if not running_under_virtualenv():
            return True
        return path.startswith(self._normalize_path_cached(sys.prefix))

    def add(self, path: str) -> None:
        head, tail = os.path.split(path)
        path = os.path.join(self._normalize_path_cached(head), os.path.normcase(tail))
        if not os.path.exists(path):
            return
        if self._permitted(path):
            self._paths.add(path)
        else:
            self._refuse.add(path)
        if os.path.splitext(path)[1] == '.py':
            self.add(cache_from_source(path))

    def add_pth(self, pth_file: str, entry: str) -> None:
        pth_file = self._normalize_path_cached(pth_file)
        if self._permitted(pth_file):
            if pth_file not in self._pth:
                self._pth[pth_file] = UninstallPthEntries(pth_file)
            self._pth[pth_file].add(entry)
        else:
            self._refuse.add(pth_file)

    def remove(self, auto_confirm: bool=False, verbose: bool=False) -> None:
        """Remove paths in ``self._paths`` with confirmation (unless
        ``auto_confirm`` is True)."""
        if not self._paths:
            logger.info("Can't uninstall '%s'. No files were found to uninstall.", self._dist.raw_name)
            return
        dist_name_version = f'{self._dist.raw_name}-{self._dist.version}'
        logger.info('Uninstalling %s:', dist_name_version)
        with indent_log():
            if auto_confirm or self._allowed_to_proceed(verbose):
                moved = self._moved_paths
                for_rename = compress_for_rename(self._paths)
                for path in sorted(compact(for_rename)):
                    moved.stash(path)
                    logger.verbose('Removing file or directory %s', path)
                for pth in self._pth.values():
                    pth.remove()
                logger.info('Successfully uninstalled %s', dist_name_version)

    def _allowed_to_proceed(self, verbose: bool) -> bool:
        """Display which files would be deleted and prompt for confirmation"""

        def _display(msg: str, paths: Iterable[str]) -> None:
            if not paths:
                return
            logger.info(msg)
            with indent_log():
                for path in sorted(compact(paths)):
                    logger.info(path)
        if not verbose:
            will_remove, will_skip = compress_for_output_listing(self._paths)
        else:
            will_remove = set(self._paths)
            will_skip = set()
        _display('Would remove:', will_remove)
        _display('Would not remove (might be manually added):', will_skip)
        _display('Would not remove (outside of prefix):', self._refuse)
        if verbose:
            _display('Will actually move:', compress_for_rename(self._paths))
        return ask('Proceed (Y/n)? ', ('y', 'n', '')) != 'n'

    def rollback(self) -> None:
        """Rollback the changes previously made by remove()."""
        if not self._moved_paths.can_rollback:
            logger.error("Can't roll back %s; was not uninstalled", self._dist.raw_name)
            return
        logger.info('Rolling back uninstall of %s', self._dist.raw_name)
        self._moved_paths.rollback()
        for pth in self._pth.values():
            pth.rollback()

    def commit(self) -> None:
        """Remove temporary save dir: rollback will no longer be possible."""
        self._moved_paths.commit()

    @classmethod
    def from_dist(cls, dist: BaseDistribution) -> 'UninstallPathSet':
        dist_location = dist.location
        info_location = dist.info_location
        if dist_location is None:
            logger.info('Not uninstalling %s since it is not installed', dist.canonical_name)
            return cls(dist)
        normalized_dist_location = normalize_path(dist_location)
        if not dist.local:
            logger.info('Not uninstalling %s at %s, outside environment %s', dist.canonical_name, normalized_dist_location, sys.prefix)
            return cls(dist)
        if normalized_dist_location in {p for p in {sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')} if p}:
            logger.info('Not uninstalling %s at %s, as it is in the standard library.', dist.canonical_name, normalized_dist_location)
            return cls(dist)
        paths_to_remove = cls(dist)
        develop_egg_link = egg_link_path_from_location(dist.raw_name)
        setuptools_flat_installation = dist.installed_with_setuptools_egg_info and info_location is not None and os.path.exists(info_location) and (not info_location.endswith(f'{dist.setuptools_filename}.egg-info'))
        if setuptools_flat_installation:
            if info_location is not None:
                paths_to_remove.add(info_location)
            installed_files = dist.iter_declared_entries()
            if installed_files is not None:
                for installed_file in installed_files:
                    paths_to_remove.add(os.path.join(dist_location, installed_file))
            elif dist.is_file('top_level.txt'):
                try:
                    namespace_packages = dist.read_text('namespace_packages.txt')
                except FileNotFoundError:
                    namespaces = []
                else:
                    namespaces = namespace_packages.splitlines(keepends=False)
                for top_level_pkg in [p for p in dist.read_text('top_level.txt').splitlines() if p and p not in namespaces]:
                    path = os.path.join(dist_location, top_level_pkg)
                    paths_to_remove.add(path)
                    paths_to_remove.add(f'{path}.py')
                    paths_to_remove.add(f'{path}.pyc')
                    paths_to_remove.add(f'{path}.pyo')
        elif dist.installed_by_distutils:
            raise UninstallationError('Cannot uninstall {!r}. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.'.format(dist.raw_name))
        elif dist.installed_as_egg:
            paths_to_remove.add(dist_location)
            easy_install_egg = os.path.split(dist_location)[1]
            easy_install_pth = os.path.join(os.path.dirname(dist_location), 'easy-install.pth')
            paths_to_remove.add_pth(easy_install_pth, './' + easy_install_egg)
        elif dist.installed_with_dist_info:
            for path in uninstallation_paths(dist):
                paths_to_remove.add(path)
        elif develop_egg_link:
            with open(develop_egg_link) as fh:
                link_pointer = os.path.normcase(fh.readline().strip())
                normalized_link_pointer = paths_to_remove._normalize_path_cached(link_pointer)
            assert os.path.samefile(normalized_link_pointer, normalized_dist_location), f'Egg-link {develop_egg_link} (to {link_pointer}) does not match installed location of {dist.raw_name} (at {dist_location})'
            paths_to_remove.add(develop_egg_link)
            easy_install_pth = os.path.join(os.path.dirname(develop_egg_link), 'easy-install.pth')
            paths_to_remove.add_pth(easy_install_pth, dist_location)
        else:
            logger.debug('Not sure how to uninstall: %s - Check: %s', dist, dist_location)
        if dist.in_usersite:
            bin_dir = get_bin_user()
        else:
            bin_dir = get_bin_prefix()
        try:
            for script in dist.iter_distutils_script_names():
                paths_to_remove.add(os.path.join(bin_dir, script))
                if WINDOWS:
                    paths_to_remove.add(os.path.join(bin_dir, f'{script}.bat'))
        except (FileNotFoundError, NotADirectoryError):
            pass

        def iter_scripts_to_remove(dist: BaseDistribution, bin_dir: str) -> Generator[str, None, None]:
            for entry_point in dist.iter_entry_points():
                if entry_point.group == 'console_scripts':
                    yield from _script_names(bin_dir, entry_point.name, False)
                elif entry_point.group == 'gui_scripts':
                    yield from _script_names(bin_dir, entry_point.name, True)
        for s in iter_scripts_to_remove(dist, bin_dir):
            paths_to_remove.add(s)
        return paths_to_remove