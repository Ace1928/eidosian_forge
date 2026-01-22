import argparse
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from shutil import which
from typing import Any, Dict, List, Tuple
import torch
from ..commands.config.config_args import SageMakerConfig
from ..utils import (
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import is_port_in_use, merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType
def _filter_args(args, parser, default_args=[]):
    """
    Filters out all `accelerate` specific args
    """
    new_args, _ = parser.parse_known_args(default_args)
    for key, value in vars(args).items():
        if key in vars(new_args).keys():
            setattr(new_args, key, value)
    return new_args