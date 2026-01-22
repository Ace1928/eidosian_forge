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
def _check_core_extension(self, extension, info, check_installed_only):
    """Check if a core extension is enabled or disabled"""
    if extension in info['uninstalled_core']:
        self.logger.info(f'{extension}:{RED_X}')
        return False
    if check_installed_only:
        self.logger.info(f'{extension}: {GREEN_OK}')
        return True
    if extension in info['disabled_core']:
        self.logger.info(f'{extension}: {RED_DISABLED}')
        return False
    self.logger.info(f'{extension}:{GREEN_ENABLED}')
    return True