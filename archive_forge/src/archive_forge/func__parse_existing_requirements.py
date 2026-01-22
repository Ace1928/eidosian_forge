import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def _parse_existing_requirements(launch_project: LaunchProject) -> str:
    import pkg_resources
    requirements_line = ''
    assert launch_project.project_dir is not None
    base_requirements = os.path.join(launch_project.project_dir, 'requirements.txt')
    if os.path.exists(base_requirements):
        include_only = set()
        with open(base_requirements) as f:
            iter = pkg_resources.parse_requirements(f)
            while True:
                try:
                    pkg = next(iter)
                    if hasattr(pkg, 'name'):
                        name = pkg.name.lower()
                    else:
                        name = str(pkg)
                    include_only.add(shlex_quote(name))
                except StopIteration:
                    break
                except Exception as e:
                    _logger.warn(f'Unable to parse requirements.txt: {e}')
                    continue
        requirements_line += 'WANDB_ONLY_INCLUDE={} '.format(','.join(include_only))
    return requirements_line