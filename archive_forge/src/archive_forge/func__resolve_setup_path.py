from distutils.util import convert_path
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import glob
import io
from setuptools.command.easy_install import easy_install
from setuptools import _path
from setuptools import namespaces
import setuptools
@staticmethod
def _resolve_setup_path(egg_base, install_dir, egg_path):
    """
        Generate a path from egg_base back to '.' where the
        setup script resides and ensure that path points to the
        setup path from $install_dir/$egg_path.
        """
    path_to_setup = egg_base.replace(os.sep, '/').rstrip('/')
    if path_to_setup != os.curdir:
        path_to_setup = '../' * (path_to_setup.count('/') + 1)
    resolved = _path.normpath(os.path.join(install_dir, egg_path, path_to_setup))
    curdir = _path.normpath(os.curdir)
    if resolved != curdir:
        raise DistutilsOptionError("Can't get a consistent path to setup script from installation directory", resolved, curdir)
    return path_to_setup