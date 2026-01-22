import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def _parse_req(req: List[str]) -> Dict[str, str]:
    d: Dict[str, str] = dict()
    for line in req:
        _name: str = None
        _version: str = None
        if line.startswith('#'):
            continue
        elif 'git+' in line or 'hg+' in line:
            _name = line.split('#egg=')[1]
            _version = line.split('@')[-1].split('#')[0]
        elif '==' in line:
            _s = line.split('==')
            _name = _s[0].lower()
            _version = _s[1].split('#')[0].strip()
        elif '>=' in line:
            _s = line.split('>=')
            _name = _s[0].lower()
            _version = _s[1].split('#')[0].strip()
        elif '>' in line:
            _s = line.split('>')
            _name = _s[0].lower()
            _version = _s[1].split('#')[0].strip()
        elif re.match(_VALID_PIP_PACKAGE_REGEX, line) is not None:
            _name = line
        else:
            raise ValueError(f'Unable to parse pip requirements file line: {line}')
        if _name is not None:
            assert re.match(_VALID_PIP_PACKAGE_REGEX, _name), f'Invalid pip package name {_name}'
            d[_name] = _version
    return d