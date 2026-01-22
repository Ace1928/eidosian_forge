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