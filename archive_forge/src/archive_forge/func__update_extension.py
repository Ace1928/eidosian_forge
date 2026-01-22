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
def _update_extension(self, name):
    """Update an extension by name.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
    data = self.info['extensions'][name]
    if data['alias_package_source']:
        self.logger.warning("Skipping updating pinned extension '%s'." % name)
        return False
    try:
        latest = self._latest_compatible_package_version(name)
    except URLError:
        return False
    if latest is None:
        self.logger.warning(f'No compatible version found for {name}!')
        return False
    if latest == data['version']:
        self.logger.info('Extension %r already up to date' % name)
        return False
    self.logger.info(f'Updating {name} to version {latest}')
    return self.install_extension(f'{name}@{latest}')