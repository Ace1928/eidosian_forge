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
def tpu_launcher(args):
    import torch_xla.distributed.xla_multiprocessing as xmp
    if args.no_python:
        raise ValueError('--no_python cannot be used with TPU launcher')
    args, current_env = prepare_tpu(args, {})
    if args.module:
        mod_name = args.training_script
    else:
        script_path = Path(args.training_script)
        sys.path.append(str(script_path.parent.resolve()))
        mod_name = script_path.stem
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, args.main_training_function):
        raise ValueError(f'Your training script should have a function named {args.main_training_function}, or you should pass a different value to `--main_training_function`.')
    sys.argv = [mod.__file__] + args.training_script_args
    main_function = getattr(mod, args.main_training_function)
    with patch_environment(**current_env):
        xmp.spawn(PrepareForLaunch(main_function), args=(), nprocs=args.num_processes)