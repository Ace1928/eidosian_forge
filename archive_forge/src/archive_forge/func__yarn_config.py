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
def _yarn_config(logger):
    """Get the yarn configuration.

    Returns
    -------
    {"yarn config": dict, "npm config": dict} if unsuccessfull the subdictionary are empty
    """
    configuration = {'yarn config': {}, 'npm config': {}}
    try:
        node = which('node')
    except ValueError:
        logger.debug('NodeJS was not found. Yarn user configuration is ignored.')
        return configuration
    try:
        output_binary = subprocess.check_output([node, YARN_PATH, 'config', '--json'], stderr=subprocess.PIPE, cwd=HERE)
        output = output_binary.decode('utf-8')
        lines = iter(output.splitlines())
        try:
            for line in lines:
                info = json.loads(line)
                if info['type'] == 'info':
                    key = info['data']
                    inspect = json.loads(next(lines))
                    if inspect['type'] == 'inspect':
                        configuration[key] = inspect['data']
        except StopIteration:
            pass
        logger.debug('Yarn configuration loaded.')
    except subprocess.CalledProcessError as e:
        logger.error('Fail to get yarn configuration. {!s}{!s}'.format(e.stderr.decode('utf-8'), e.output.decode('utf-8')))
    except Exception as e:
        logger.error(f'Fail to get yarn configuration. {e!s}')
    return configuration