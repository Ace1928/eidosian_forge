import importlib
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from os.path import basename, normpath
from os.path import join as pjoin
from jupyter_core.paths import ENV_JUPYTER_PATH, SYSTEM_JUPYTER_PATH, jupyter_data_dir
from jupyter_core.utils import ensure_dir_exists
from jupyter_server.extension.serverextension import ArgumentConflict
from jupyterlab_server.config import get_federated_extensions
from .commands import _test_overlap
def _get_labextension_metadata(module):
    """Get the list of labextension paths associated with a Python module.

    Returns a tuple of (the module path,             [{
        'src': 'mockextension',
        'dest': '_mockdestination'
    }])

    Parameters
    ----------

    module : str
        Importable Python module exposing the
        magic-named `_jupyter_labextension_paths` function
    """
    mod_path = osp.abspath(module)
    if not osp.exists(mod_path):
        msg = f'The path `{mod_path}` does not exist.'
        raise FileNotFoundError(msg)
    errors = []
    try:
        m = importlib.import_module(module)
        if hasattr(m, '_jupyter_labextension_paths'):
            return (m, m._jupyter_labextension_paths())
    except Exception as exc:
        errors.append(exc)
    package = None
    if os.path.exists(os.path.join(mod_path, 'pyproject.toml')):
        with open(os.path.join(mod_path, 'pyproject.toml'), 'rb') as fid:
            data = load(fid)
        package = data.get('project', {}).get('name')
    if not package:
        try:
            package = subprocess.check_output([sys.executable, 'setup.py', '--name'], cwd=mod_path).decode('utf8').strip()
        except subprocess.CalledProcessError:
            msg = f'The Python package `{module}` is not a valid package, it is missing the `setup.py` file.'
            raise FileNotFoundError(msg) from None
    try:
        version(package)
    except PackageNotFoundError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', mod_path])
        sys.path.insert(0, mod_path)
    from setuptools import find_namespace_packages, find_packages
    package_candidates = [package.replace('-', '_')]
    package_candidates.extend(find_packages(mod_path))
    package_candidates.extend(find_namespace_packages(mod_path))
    for package in package_candidates:
        try:
            m = importlib.import_module(package)
            if hasattr(m, '_jupyter_labextension_paths'):
                return (m, m._jupyter_labextension_paths())
        except Exception as exc:
            errors.append(exc)
    msg = f'There is no labextension at {module}. Errors encountered: {errors}'
    raise ModuleNotFoundError(msg)