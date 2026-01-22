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
def _create_docker_build_ctx(launch_project: LaunchProject, dockerfile_contents: str) -> str:
    """Create a build context temp dir for a Dockerfile and project code."""
    assert launch_project.project_dir is not None
    directory = tempfile.mkdtemp()
    entrypoint = launch_project.get_single_entry_point()
    if entrypoint is not None:
        assert entrypoint.name is not None
        entrypoint_dir = os.path.dirname(entrypoint.name)
        if entrypoint_dir:
            path = os.path.join(launch_project.project_dir, entrypoint_dir, _WANDB_DOCKERFILE_NAME)
        else:
            path = os.path.join(launch_project.project_dir, _WANDB_DOCKERFILE_NAME)
        if os.path.exists(path):
            shutil.copytree(os.path.dirname(path), directory, symlinks=True, dirs_exist_ok=True, ignore=shutil.ignore_patterns('fsmonitor--daemon.ipc'))
            if entrypoint_dir:
                new_path = os.path.basename(entrypoint.name)
                entrypoint = launch_project.get_single_entry_point()
                if entrypoint is not None:
                    entrypoint.update_entrypoint_path(new_path)
            return directory
    dst_path = os.path.join(directory, 'src')
    assert launch_project.project_dir is not None
    shutil.copytree(src=launch_project.project_dir, dst=dst_path, symlinks=True, ignore=shutil.ignore_patterns('fsmonitor--daemon.ipc'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'templates', '_wandb_bootstrap.py'), os.path.join(directory))
    if launch_project.python_version:
        runtime_path = os.path.join(dst_path, 'runtime.txt')
        with open(runtime_path, 'w') as fp:
            fp.write(f'python-{launch_project.python_version}')
    with open(os.path.join(directory, _WANDB_DOCKERFILE_NAME), 'w') as handle:
        handle.write(dockerfile_contents)
    return directory