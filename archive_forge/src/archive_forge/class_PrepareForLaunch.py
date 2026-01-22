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
class PrepareForLaunch:
    """
    Prepare a function that will launched in a distributed setup.

    Args:
        launcher (`Callable`):
            The function to launch.
        distributed_type ([`~state.DistributedType`]):
            The distributed type to prepare for.
        debug (`bool`, *optional*, defaults to `False`):
            Whether or not this is a debug launch.
    """

    def __init__(self, launcher, distributed_type='NO', debug=False):
        self.launcher = launcher
        self.distributed_type = DistributedType(distributed_type)
        self.debug = debug

    def __call__(self, index, *args):
        if self.debug:
            world_size = int(os.environ.get('WORLD_SIZE'))
            rdv_file = os.environ.get('ACCELERATE_DEBUG_RDV_FILE')
            torch.distributed.init_process_group('gloo', rank=index, store=torch.distributed.FileStore(rdv_file, world_size), world_size=world_size)
        elif self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_NPU, DistributedType.MULTI_XPU, DistributedType.MULTI_CPU):
            os.environ['LOCAL_RANK'] = str(index)
            nproc = int(os.environ.get('NPROC', 1))
            node_rank = int(os.environ.get('NODE_RANK', 0))
            os.environ['RANK'] = str(nproc * node_rank + index)
        os.environ['FORK_LAUNCHED'] = str(1)
        self.launcher(*args)