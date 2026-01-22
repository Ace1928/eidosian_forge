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
def _convert_nargs_to_dict(nargs: List[str]) -> Dict[str, str]:
    if len(nargs) < 0:
        return {}

    def _infer_type(s):
        try:
            s = float(s)
            if s // 1 == s:
                return int(s)
            return s
        except ValueError:
            return s
    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args(nargs)
    for index, argument in enumerate(unknown):
        if argument.startswith(('-', '--')):
            action = None
            if index + 1 < len(unknown):
                if unknown[index + 1].startswith(('-', '--')):
                    raise ValueError('SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types')
            else:
                raise ValueError('SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types')
            if action is None:
                parser.add_argument(argument, type=_infer_type)
            else:
                parser.add_argument(argument, action=action)
    return {key: literal_eval(value) if value in ('True', 'False') else value for key, value in parser.parse_args(nargs).__dict__.items()}