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
def _is_wandb_uri(uri: str) -> bool:
    return (_WANDB_URI_REGEX.match(uri) or _WANDB_DEV_URI_REGEX.match(uri) or _WANDB_LOCAL_DEV_URI_REGEX.match(uri) or _WANDB_QA_URI_REGEX.match(uri)) is not None