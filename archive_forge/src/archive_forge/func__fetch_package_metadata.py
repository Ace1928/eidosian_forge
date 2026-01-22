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
def _fetch_package_metadata(registry, name, logger):
    """Fetch the metadata for a package from the npm registry"""
    req = Request(urljoin(registry, quote(name, safe='@')), headers={'Accept': 'application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8, */*'})
    try:
        logger.debug('Fetching URL: %s' % req.full_url)
    except AttributeError:
        logger.debug('Fetching URL: %s' % req.get_full_url())
    try:
        with contextlib.closing(urlopen(req)) as response:
            return json.loads(response.read().decode('utf-8'))
    except URLError as exc:
        logger.warning('Failed to fetch package metadata for %r: %r', name, exc)
        raise