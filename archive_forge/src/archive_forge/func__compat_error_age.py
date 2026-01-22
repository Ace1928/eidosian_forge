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
def _compat_error_age(errors):
    """Compare all incompatibilities for an extension.

    Returns a number > 0 if all extensions are older than that supported by lab.
    Returns a number < 0 if all extensions are newer than that supported by lab.
    Returns 0 otherwise (i.e. a mix).
    """
    any_older = False
    any_newer = False
    for _, jlab, ext in errors:
        c = _compare_ranges(ext, jlab, drop_prerelease1=True)
        any_newer = any_newer or c < 0
        any_older = any_older or c > 0
    if any_older and (not any_newer):
        return 1
    elif any_newer and (not any_older):
        return -1
    return 0