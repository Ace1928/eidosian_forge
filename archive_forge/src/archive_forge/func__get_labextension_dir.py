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
def _get_labextension_dir(user=False, sys_prefix=False, prefix=None, labextensions_dir=None):
    """Return the labextension directory specified

    Parameters
    ----------

    user : bool [default: False]
        Get the user's .jupyter/labextensions directory
    sys_prefix : bool [default: False]
        Get sys.prefix, i.e. ~/.envs/my-env/share/jupyter/labextensions
    prefix : str [optional]
        Get custom prefix
    labextensions_dir : str [optional]
        Get what you put in
    """
    conflicting = [('user', user), ('prefix', prefix), ('labextensions_dir', labextensions_dir), ('sys_prefix', sys_prefix)]
    conflicting_set = [f'{n}={v!r}' for n, v in conflicting if v]
    if len(conflicting_set) > 1:
        msg = 'cannot specify more than one of user, sys_prefix, prefix, or labextensions_dir, but got: {}'.format(', '.join(conflicting_set))
        raise ArgumentConflict(msg)
    if user:
        labext = pjoin(jupyter_data_dir(), 'labextensions')
    elif sys_prefix:
        labext = pjoin(ENV_JUPYTER_PATH[0], 'labextensions')
    elif prefix:
        labext = pjoin(prefix, 'share', 'jupyter', 'labextensions')
    elif labextensions_dir:
        labext = labextensions_dir
    else:
        labext = pjoin(SYSTEM_JUPYTER_PATH[0], 'labextensions')
    return labext