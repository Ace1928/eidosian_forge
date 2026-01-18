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
def watch_labextension(path, labextensions_path, logger=None, development=False, source_map=False, core_path=None):
    """Watch a labextension in a given path"""
    core_path = osp.join(HERE, 'staging') if core_path is None else str(Path(core_path).resolve())
    ext_path = str(Path(path).resolve())
    if logger:
        logger.info('Building extension in %s' % path)
    federated_extensions = get_federated_extensions(labextensions_path)
    with open(pjoin(ext_path, 'package.json')) as fid:
        ext_data = json.load(fid)
    if ext_data['name'] not in federated_extensions:
        develop_labextension_py(ext_path, sys_prefix=True)
    else:
        full_dest = pjoin(federated_extensions[ext_data['name']]['ext_dir'], ext_data['name'])
        output_dir = pjoin(ext_path, ext_data['jupyterlab'].get('outputDir', 'static'))
        if not osp.islink(full_dest):
            shutil.rmtree(full_dest)
            os.symlink(output_dir, full_dest)
    builder = _ensure_builder(ext_path, core_path)
    arguments = ['node', builder, '--core-path', core_path, '--watch', ext_path]
    if development:
        arguments.append('--development')
    if source_map:
        arguments.append('--source-map')
    subprocess.check_call(arguments, cwd=ext_path)