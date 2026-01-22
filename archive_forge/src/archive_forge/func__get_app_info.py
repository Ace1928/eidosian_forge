import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
def _get_app_info(self):
    """Get information about the app."""
    info = {}
    info['core_data'] = core_data = self.core_data
    info['extensions'] = extensions = self._get_extensions(core_data)
    info['local_extensions'] = self._get_local_extensions()
    info['linked_packages'] = self._get_linked_packages()
    info['app_extensions'] = app = []
    info['sys_extensions'] = sys = []
    for name, data in extensions.items():
        data['is_local'] = name in info['local_extensions']
        if data['location'] == 'app':
            app.append(name)
        else:
            sys.append(name)
    info['uninstalled_core'] = self._get_uninstalled_core_extensions()
    info['static_data'] = _get_static_data(self.app_dir)
    app_data = info['static_data'] or core_data
    info['version'] = app_data['jupyterlab']['version']
    info['staticUrl'] = app_data['jupyterlab'].get('staticUrl', '')
    info['sys_dir'] = self.sys_dir
    info['app_dir'] = self.app_dir
    info['core_extensions'] = _get_core_extensions(self.core_data)
    info['federated_extensions'] = get_federated_extensions(self.labextensions_path)
    info['shadowed_exts'] = [ext for ext in info['extensions'] if ext in info['federated_extensions']]
    return info