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
def _get_extensions_in_dir(self, dname, core_data):
    """Get the extensions in a given directory."""
    extensions = {}
    location = 'app' if dname == self.app_dir else 'sys'
    for target in glob(pjoin(dname, 'extensions', '*.tgz')):
        data = read_package(target)
        deps = data.get('dependencies', {})
        name = data['name']
        jlab = data.get('jupyterlab', {})
        path = osp.abspath(target)
        filename = osp.basename(target)
        if filename.startswith(PIN_PREFIX):
            alias = filename[len(PIN_PREFIX):-len('.tgz')]
        else:
            alias = None
        url = get_package_url(data)
        extensions[alias or name] = {'description': data.get('description', ''), 'path': path, 'filename': osp.basename(path), 'url': url, 'version': data['version'], 'alias_package_source': name if alias else None, 'jupyterlab': jlab, 'dependencies': deps, 'tar_dir': osp.dirname(path), 'location': location}
    return extensions