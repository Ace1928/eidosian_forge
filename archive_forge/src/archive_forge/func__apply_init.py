import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
def _apply_init(self, init_settings: Dict[str, Union[str, int, None]]) -> None:
    init_settings.pop('magic', None)
    if self.sweep_id:
        for key in ('project', 'entity', 'id'):
            val = init_settings.pop(key, None)
            if val:
                wandb.termwarn(f'Ignored wandb.init() arg {key} when running a sweep.')
    if self.launch:
        if self.project is not None and init_settings.pop('project', None):
            wandb.termwarn('Project is ignored when running from wandb launch context. Ignored wandb.init() arg project when running running from launch.')
        for key in ('entity', 'id'):
            if init_settings.pop(key, None):
                wandb.termwarn(f'Project, entity and id are ignored when running from wandb launch context. Ignored wandb.init() arg {key} when running running from launch.')
    param_map = dict(name='run_name', id='run_id', tags='run_tags', group='run_group', job_type='run_job_type', notes='run_notes', dir='root_dir', sweep_id='sweep_id')
    init_settings = {param_map.get(k, k): v for k, v in init_settings.items() if v is not None}
    if init_settings.get('resume'):
        if isinstance(init_settings['resume'], str):
            if init_settings['resume'] not in ('allow', 'must', 'never', 'auto'):
                if init_settings.get('run_id') is None:
                    init_settings['run_id'] = init_settings['resume']
                init_settings['resume'] = 'allow'
        elif init_settings['resume'] is True:
            init_settings['resume'] = 'auto'
    self.update(init_settings, source=Source.INIT)
    if self.resume == 'auto':
        if os.path.exists(self.resume_fname):
            with open(self.resume_fname) as f:
                resume_run_id = json.load(f)['run_id']
            if self.run_id is None:
                self.update({'run_id': resume_run_id}, source=Source.INIT)
            elif self.run_id != resume_run_id:
                wandb.termwarn(f'Tried to auto resume run with id {resume_run_id} but id {self.run_id} is set.')
    self.update({'run_id': self.run_id or generate_id()}, source=Source.INIT)
    if self.resume == 'auto' and self.resume_fname is not None:
        filesystem.mkdir_exists_ok(self.wandb_dir)
        with open(self.resume_fname, 'w') as f:
            f.write(json.dumps({'run_id': self.run_id}))