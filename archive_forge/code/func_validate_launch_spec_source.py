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
def validate_launch_spec_source(launch_spec: Dict[str, Any]) -> None:
    uri = launch_spec.get('uri')
    job = launch_spec.get('job')
    docker_image = launch_spec.get('docker', {}).get('docker_image')
    if not bool(uri) and (not bool(job)) and (not bool(docker_image)):
        raise LaunchError('Must specify a uri, job or docker image')
    elif bool(uri) and bool(docker_image):
        raise LaunchError('Found both uri and docker-image, only one can be set')
    elif sum(map(bool, [uri, job, docker_image])) > 1:
        raise LaunchError('Must specify exactly one of uri, job or image')