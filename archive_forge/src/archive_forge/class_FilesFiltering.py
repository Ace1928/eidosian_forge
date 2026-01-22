import fnmatch
import glob
import os.path
import sys
from _pydev_bundle import pydev_log
import pydevd_file_utils
import json
from collections import namedtuple
from _pydev_bundle._pydev_saved_modules import threading
from pydevd_file_utils import normcase
from _pydevd_bundle.pydevd_constants import USER_CODE_BASENAMES_STARTING_WITH, \
from _pydevd_bundle import pydevd_constants
class FilesFiltering(object):
    """
    Note: calls at FilesFiltering are uncached.

    The actual API used should be through PyDB.
    """

    def __init__(self):
        self._exclude_filters = []
        self._project_roots = []
        self._library_roots = []
        self._use_libraries_filter = False
        self.require_module = False
        self.set_use_libraries_filter(os.getenv('PYDEVD_FILTER_LIBRARIES') is not None)
        project_roots = os.getenv('IDE_PROJECT_ROOTS', None)
        if project_roots is not None:
            project_roots = project_roots.split(os.pathsep)
        else:
            project_roots = []
        self.set_project_roots(project_roots)
        library_roots = os.getenv('LIBRARY_ROOTS', None)
        if library_roots is not None:
            library_roots = library_roots.split(os.pathsep)
        else:
            library_roots = self._get_default_library_roots()
        self.set_library_roots(library_roots)
        pydevd_filters = os.getenv('PYDEVD_FILTERS', '')
        if pydevd_filters:
            pydev_log.debug('PYDEVD_FILTERS %s', (pydevd_filters,))
            if pydevd_filters.startswith('{'):
                exclude_filters = []
                for key, val in json.loads(pydevd_filters).items():
                    exclude_filters.append(ExcludeFilter(key, val, True))
                self._exclude_filters = exclude_filters
            else:
                filters = pydevd_filters.split(';')
                new_filters = []
                for new_filter in filters:
                    if new_filter.strip():
                        new_filters.append(ExcludeFilter(new_filter.strip(), True, True))
                self._exclude_filters = new_filters

    @classmethod
    def _get_default_library_roots(cls):
        pydev_log.debug('Collecting default library roots.')
        import site
        roots = []
        try:
            import sysconfig
        except ImportError:
            pass
        else:
            for path_name in set(('stdlib', 'platstdlib', 'purelib', 'platlib')) & set(sysconfig.get_path_names()):
                roots.append(sysconfig.get_path(path_name))
        roots.append(os.path.dirname(os.__file__))
        roots.append(os.path.dirname(threading.__file__))
        if IS_PYPY:
            try:
                import _pypy_wait
            except ImportError:
                pydev_log.debug('Unable to import _pypy_wait on PyPy when collecting default library roots.')
            else:
                pypy_lib_dir = os.path.dirname(_pypy_wait.__file__)
                pydev_log.debug('Adding %s to default library roots.', pypy_lib_dir)
                roots.append(pypy_lib_dir)
        if hasattr(site, 'getusersitepackages'):
            site_paths = site.getusersitepackages()
            if isinstance(site_paths, (list, tuple)):
                for site_path in site_paths:
                    roots.append(site_path)
            else:
                roots.append(site_paths)
        if hasattr(site, 'getsitepackages'):
            site_paths = site.getsitepackages()
            if isinstance(site_paths, (list, tuple)):
                for site_path in site_paths:
                    roots.append(site_path)
            else:
                roots.append(site_paths)
        for path in sys.path:
            if os.path.exists(path) and os.path.basename(path) in ('site-packages', 'pip-global'):
                roots.append(path)
        roots = [path for path in roots if path is not None]
        roots.extend([os.path.realpath(path) for path in roots])
        return sorted(set(roots))

    def _fix_roots(self, roots):
        roots = _convert_to_str_and_clear_empty(roots)
        new_roots = []
        for root in roots:
            path = self._absolute_normalized_path(root)
            if pydevd_constants.IS_WINDOWS:
                new_roots.append(path + '\\')
            else:
                new_roots.append(path + '/')
        return new_roots

    def _absolute_normalized_path(self, filename):
        """
        Provides a version of the filename that's absolute and normalized.
        """
        return normcase(pydevd_file_utils.absolute_path(filename))

    def set_project_roots(self, project_roots):
        self._project_roots = self._fix_roots(project_roots)
        pydev_log.debug('IDE_PROJECT_ROOTS %s\n' % project_roots)

    def _get_project_roots(self):
        return self._project_roots

    def set_library_roots(self, roots):
        self._library_roots = self._fix_roots(roots)
        pydev_log.debug('LIBRARY_ROOTS %s\n' % roots)

    def _get_library_roots(self):
        return self._library_roots

    def in_project_roots(self, received_filename):
        """
        Note: don't call directly. Use PyDb.in_project_scope (there's no caching here and it doesn't
        handle all possibilities for knowing whether a project is actually in the scope, it
        just handles the heuristics based on the absolute_normalized_filename without the actual frame).
        """
        DEBUG = False
        if received_filename.startswith(USER_CODE_BASENAMES_STARTING_WITH):
            if DEBUG:
                pydev_log.debug('In in_project_roots - user basenames - starts with %s (%s)', received_filename, USER_CODE_BASENAMES_STARTING_WITH)
            return True
        if received_filename.startswith(LIBRARY_CODE_BASENAMES_STARTING_WITH):
            if DEBUG:
                pydev_log.debug('Not in in_project_roots - library basenames - starts with %s (%s)', received_filename, LIBRARY_CODE_BASENAMES_STARTING_WITH)
            return False
        project_roots = self._get_project_roots()
        absolute_normalized_filename = self._absolute_normalized_path(received_filename)
        absolute_normalized_filename_as_dir = absolute_normalized_filename + ('\\' if IS_WINDOWS else '/')
        found_in_project = []
        for root in project_roots:
            if root and (absolute_normalized_filename.startswith(root) or root == absolute_normalized_filename_as_dir):
                if DEBUG:
                    pydev_log.debug('In project: %s (%s)', absolute_normalized_filename, root)
                found_in_project.append(root)
        found_in_library = []
        library_roots = self._get_library_roots()
        for root in library_roots:
            if root and (absolute_normalized_filename.startswith(root) or root == absolute_normalized_filename_as_dir):
                found_in_library.append(root)
                if DEBUG:
                    pydev_log.debug('In library: %s (%s)', absolute_normalized_filename, root)
            elif DEBUG:
                pydev_log.debug('Not in library: %s (%s)', absolute_normalized_filename, root)
        if not project_roots:
            in_project = not found_in_library
            if DEBUG:
                pydev_log.debug('Final in project (no project roots): %s (%s)', absolute_normalized_filename, in_project)
        else:
            in_project = False
            if found_in_project:
                if not found_in_library:
                    if DEBUG:
                        pydev_log.debug('Final in project (in_project and not found_in_library): %s (True)', absolute_normalized_filename)
                    in_project = True
                else:
                    if max((len(x) for x in found_in_project)) > max((len(x) for x in found_in_library)):
                        in_project = True
                    if DEBUG:
                        pydev_log.debug('Final in project (found in both): %s (%s)', absolute_normalized_filename, in_project)
        return in_project

    def use_libraries_filter(self):
        """
        Should we debug only what's inside project folders?
        """
        return self._use_libraries_filter

    def set_use_libraries_filter(self, use):
        pydev_log.debug('pydevd: Use libraries filter: %s\n' % use)
        self._use_libraries_filter = use

    def use_exclude_filters(self):
        return len(self._exclude_filters) > 0

    def exclude_by_filter(self, absolute_filename, module_name):
        """
        :return: True if it should be excluded, False if it should be included and None
            if no rule matched the given file.
        """
        for exclude_filter in self._exclude_filters:
            if exclude_filter.is_path:
                if glob_matches_path(absolute_filename, exclude_filter.name):
                    return exclude_filter.exclude
            elif exclude_filter.name == module_name or module_name.startswith(exclude_filter.name + '.'):
                return exclude_filter.exclude
        return None

    def set_exclude_filters(self, exclude_filters):
        """
        :param list(ExcludeFilter) exclude_filters:
        """
        self._exclude_filters = exclude_filters
        self.require_module = False
        for exclude_filter in exclude_filters:
            if not exclude_filter.is_path:
                self.require_module = True
                break