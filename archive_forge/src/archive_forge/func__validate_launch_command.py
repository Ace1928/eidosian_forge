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
def _validate_launch_command(args):
    if sum([args.multi_gpu, args.cpu, args.tpu, args.use_deepspeed, args.use_fsdp]) > 1:
        raise ValueError('You can only use one of `--cpu`, `--multi_gpu`, `--tpu`, `--use_deepspeed`, `--use_fsdp` at a time.')
    if args.multi_gpu and args.num_processes is not None and (args.num_processes < 2):
        raise ValueError('You need to use at least 2 processes to use `--multi_gpu`.')
    defaults = None
    warned = []
    mp_from_config_flag = False
    if args.config_file is not None or (os.path.isfile(default_config_file) and (not args.cpu)):
        defaults = load_config_from_file(args.config_file)
        if not args.multi_gpu and (not args.tpu) and (not args.tpu_use_cluster) and (not args.use_deepspeed) and (not args.use_fsdp) and (not args.use_megatron_lm):
            args.use_deepspeed = defaults.distributed_type == DistributedType.DEEPSPEED
            args.multi_gpu = True if defaults.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_NPU, DistributedType.MULTI_XPU) else False
            args.tpu = defaults.distributed_type == DistributedType.XLA
            args.use_fsdp = defaults.distributed_type == DistributedType.FSDP
            args.use_megatron_lm = defaults.distributed_type == DistributedType.MEGATRON_LM
            args.tpu_use_cluster = defaults.tpu_use_cluster if args.tpu else False
        if args.gpu_ids is None:
            if defaults.gpu_ids is not None:
                args.gpu_ids = defaults.gpu_ids
            else:
                args.gpu_ids = 'all'
        if args.multi_gpu and args.num_machines is None:
            args.num_machines = defaults.num_machines
        if len(args.gpu_ids.split(',')) < 2 and args.gpu_ids != 'all' and args.multi_gpu and (args.num_machines <= 1):
            raise ValueError("Less than two GPU ids were configured and tried to run on on multiple GPUs. Please ensure at least two are specified for `--gpu_ids`, or use `--gpu_ids='all'`.")
        if defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            for name, attr in defaults.__dict__.items():
                if isinstance(attr, dict):
                    for k in defaults.deepspeed_config:
                        setattr(args, k, defaults.deepspeed_config[k])
                    for k in defaults.fsdp_config:
                        arg_to_set = k
                        if 'fsdp' not in arg_to_set:
                            arg_to_set = 'fsdp_' + arg_to_set
                        setattr(args, arg_to_set, defaults.fsdp_config[k])
                    for k in defaults.megatron_lm_config:
                        setattr(args, k, defaults.megatron_lm_config[k])
                    for k in defaults.dynamo_config:
                        setattr(args, k, defaults.dynamo_config[k])
                    for k in defaults.ipex_config:
                        setattr(args, k, defaults.ipex_config[k])
                    for k in defaults.mpirun_config:
                        setattr(args, k, defaults.mpirun_config[k])
                    continue
                if name not in ['compute_environment', 'mixed_precision', 'distributed_type'] and getattr(args, name, None) is None:
                    setattr(args, name, attr)
        if not args.debug:
            args.debug = defaults.debug
        if not args.mixed_precision:
            if defaults.mixed_precision is None:
                args.mixed_precision = 'no'
            else:
                args.mixed_precision = defaults.mixed_precision
                mp_from_config_flag = True
        else:
            if args.use_cpu or (args.use_xpu and torch.xpu.is_available()):
                native_amp = is_torch_version('>=', '1.10')
            else:
                native_amp = is_bf16_available(True)
            if args.mixed_precision == 'bf16' and (not native_amp) and (not (args.tpu and is_torch_xla_available(check_is_tpu=True))):
                raise ValueError('bf16 mixed precision requires PyTorch >= 1.10 and a supported device.')
        if args.dynamo_backend is None:
            args.dynamo_backend = 'no'
    else:
        if args.num_processes is None:
            if args.use_xpu and is_xpu_available():
                args.num_processes = torch.xpu.device_count()
            elif is_npu_available():
                args.num_processes = torch.npu.device_count()
            else:
                args.num_processes = torch.cuda.device_count()
            warned.append(f'\t`--num_processes` was set to a value of `{args.num_processes}`')
        if args.debug is None:
            args.debug = False
        if not args.multi_gpu and (args.use_xpu and is_xpu_available() and (torch.xpu.device_count() > 1) or (is_npu_available() and torch.npu.device_count() > 1) or torch.cuda.device_count() > 1):
            warned.append('\t\tMore than one GPU was found, enabling multi-GPU training.\n\t\tIf this was unintended please pass in `--num_processes=1`.')
            args.multi_gpu = True
        if args.num_machines is None:
            warned.append('\t`--num_machines` was set to a value of `1`')
            args.num_machines = 1
        if args.mixed_precision is None:
            warned.append("\t`--mixed_precision` was set to a value of `'no'`")
            args.mixed_precision = 'no'
        if not hasattr(args, 'use_cpu'):
            args.use_cpu = args.cpu
        if args.dynamo_backend is None:
            warned.append("\t`--dynamo_backend` was set to a value of `'no'`")
            args.dynamo_backend = 'no'
    if args.debug:
        logger.debug('Running script in debug mode, expect distributed operations to be slightly slower.')
    is_aws_env_disabled = defaults is None or (defaults is not None and defaults.compute_environment != ComputeEnvironment.AMAZON_SAGEMAKER)
    if is_aws_env_disabled and args.num_cpu_threads_per_process is None:
        args.num_cpu_threads_per_process = 1
        if args.use_cpu and args.num_processes >= 1:
            local_size = get_int_from_env(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
            threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
            if threads_per_process > 1:
                args.num_cpu_threads_per_process = threads_per_process
                warned.append(f'\t`--num_cpu_threads_per_process` was set to `{args.num_cpu_threads_per_process}` to improve out-of-box performance when training on CPUs')
    if any(warned):
        message = 'The following values were not passed to `accelerate launch` and had defaults used instead:\n'
        message += '\n'.join(warned)
        message += '\nTo avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.'
        logger.warning(message)
    return (args, defaults, mp_from_config_flag)