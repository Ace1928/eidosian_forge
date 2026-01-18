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
def unlink_package(self, path):
    """Unlink a package by name or at the given path.

        A ValueError is raised if the path is not an unlinkable package.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
    path = _normalize_path(path)
    config = self._read_build_config()
    linked = config.setdefault('linked_packages', {})
    found = None
    for name, source in linked.items():
        if path in {name, source}:
            found = name
    if found:
        del linked[found]
    else:
        local = config.setdefault('local_extensions', {})
        for name, source in local.items():
            if path in {name, source}:
                found = name
        if found:
            del local[found]
            path = self.info['extensions'][found]['path']
            os.remove(path)
    if not found:
        raise ValueError('No linked package for %s' % path)
    self._write_build_config(config)
    return True