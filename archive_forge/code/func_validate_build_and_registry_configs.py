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
def validate_build_and_registry_configs(build_config: Dict[str, Any], registry_config: Dict[str, Any]) -> None:
    build_config_credentials = build_config.get('credentials', {})
    registry_config_credentials = registry_config.get('credentials', {})
    if build_config_credentials and registry_config_credentials and (build_config_credentials != registry_config_credentials):
        raise LaunchError('registry and build config credential mismatch')