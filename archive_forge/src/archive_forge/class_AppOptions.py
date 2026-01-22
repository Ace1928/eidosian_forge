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
class AppOptions(HasTraits):
    """Options object for build system"""

    def __init__(self, logger=None, core_config=None, **kwargs):
        if core_config is not None:
            kwargs['core_config'] = core_config
        if logger is not None:
            kwargs['logger'] = logger
        if 'app_dir' in kwargs and (not kwargs['app_dir']):
            kwargs.pop('app_dir')
        super().__init__(**kwargs)
    app_dir = Unicode(help='The application directory')
    use_sys_dir = Bool(True, help='Whether to shadow the default app_dir if that is set to a non-default value')
    logger = Instance(logging.Logger, help='The logger to use')
    core_config = Instance(CoreConfig, help='Configuration for core data')
    kill_event = Instance(Event, args=(), help='Event for aborting call')
    labextensions_path = List(Unicode(), help='The paths to look in for prebuilt JupyterLab extensions')
    registry = Unicode(help='NPM packages registry URL')
    splice_source = Bool(False, help='Splice source packages into app directory.')
    skip_full_build_check = Bool(False, help='If true, perform only a quick check that the lab build is up to date. If false, perform a thorough check, which verifies extension contents.')

    @default('logger')
    def _default_logger(self):
        return logging.getLogger('jupyterlab')

    @default('app_dir')
    def _default_app_dir(self):
        return get_app_dir()

    @default('core_config')
    def _default_core_config(self):
        return CoreConfig()

    @default('registry')
    def _default_registry(self):
        config = _yarn_config(self.logger)['yarn config']
        return config.get('registry', YARN_DEFAULT_REGISTRY)