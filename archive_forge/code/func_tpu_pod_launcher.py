import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
def tpu_pod_launcher(args):
    from torch_xla.distributed import xla_dist
    current_env = {}
    args, current_env = prepare_tpu(args, current_env, True)
    debug = getattr(args, 'debug', False)
    training_script = args.training_script
    training_script_args = args.training_script_args
    new_args = _filter_args(args, xla_dist.get_args_parser(), ['--tpu', args.tpu_name, '--positional', '', '--restart-tpuvm-pod-server'])
    if args.tpu_use_sudo:
        new_cmd = ['sudo']
    else:
        new_cmd = []
    new_cmd += ['accelerate-launch', '--tpu', '--no_tpu_cluster', '--num_machines', '1', '--mixed_precision', 'no', '--dynamo_backend', 'no', '--num_processes', str(args.num_processes), '--main_training_function', str(args.main_training_function), training_script] + training_script_args
    new_args.positional = new_cmd
    bad_flags = ''
    for arg in vars(new_args):
        if arg.startswith('docker_'):
            value = getattr(new_args, arg)
            if value != '' and value is not None:
                bad_flags += f'{arg}="{value}"\n'
    if bad_flags != '':
        raise ValueError(f'Docker containers are not supported for TPU pod launcher currently, please remove the following flags:\n{bad_flags}')
    new_args.env = [f'{k}={v}' for k, v in current_env.items()]
    new_args.env.append('ACCELERATE_IN_TPU_POD=1')
    try:
        xla_dist.resolve_and_execute(new_args)
    except Exception:
        if is_rich_available() and debug:
            console = get_console()
            console.print('\n[bold red]Using --debug, `torch_xla.xla_dist` Stack Trace:[/bold red]')
            console.print_exception(suppress=[__file__], show_locals=False)
        else:
            raise