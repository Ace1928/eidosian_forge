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
def _compose_extra_status(self, name: str, info: dict, data: dict, errors) -> str:
    extra = ''
    if _is_disabled(name, info['disabled']):
        extra += ' %s' % RED_DISABLED
    else:
        extra += ' %s' % GREEN_ENABLED
    if errors:
        extra += ' %s' % RED_X
    else:
        extra += ' %s' % GREEN_OK
    if data['is_local']:
        extra += '*'
    lock_status = _is_locked(name, info['locked'])
    if lock_status.entire_extension_locked:
        extra += ' 🔒 (all plugins locked)'
    elif lock_status.locked_plugins:
        plugin_list = ', '.join(sorted(lock_status.locked_plugins))
        extra += ' 🔒 (plugins: %s locked)' % plugin_list
    return extra