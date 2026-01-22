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
def _ensure_disabled_info(self):
    info = self.info
    if 'disabled' in info:
        return
    labextensions_path = self.labextensions_path
    app_settings_dir = osp.join(self.app_dir, 'settings')
    page_config = get_page_config(labextensions_path, app_settings_dir=app_settings_dir, logger=self.logger)
    disabled = page_config.get('disabledExtensions', {})
    if isinstance(disabled, list):
        disabled = {extension: True for extension in disabled}
    info['disabled'] = disabled
    locked = page_config.get('lockedExtensions', {})
    if isinstance(locked, list):
        locked = {extension: True for extension in locked}
    info['locked'] = locked
    disabled_core = []
    for key in info['core_extensions']:
        if key in info['disabled']:
            disabled_core.append(key)
    info['disabled_core'] = disabled_core