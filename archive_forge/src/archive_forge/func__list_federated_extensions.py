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
def _list_federated_extensions(self):
    self._ensure_disabled_info()
    info = self.info
    logger = self.logger
    error_accumulator = {}
    ext_dirs = {p: False for p in self.labextensions_path}
    for value in info['federated_extensions'].values():
        ext_dirs[value['ext_dir']] = True
    for ext_dir, has_exts in ext_dirs.items():
        if not has_exts:
            continue
        logger.info(ext_dir)
        for name in info['federated_extensions']:
            data = info['federated_extensions'][name]
            if data['ext_dir'] != ext_dir:
                continue
            version = data['version']
            errors = info['compat_errors'][name]
            extra = self._compose_extra_status(name, info, data, errors)
            install = data.get('install')
            if install:
                extra += ' ({}, {})'.format(install['packageManager'], install['packageName'])
            logger.info(f'        {name} v{version}{extra}')
            if errors:
                error_accumulator[name] = (version, errors)
        logger.info('')
    _log_multiple_compat_errors(logger, error_accumulator)