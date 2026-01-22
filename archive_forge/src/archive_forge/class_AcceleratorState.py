from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
class AcceleratorState:
    """
    Singleton class that has information about the current training environment.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **initialized** (`bool`) -- Whether or not the `AcceleratorState` has been initialized from `Accelerator`.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed. (Choose from 'no','fp16','bf16 or 'fp8').
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
        - **debug** (`bool`) -- Whether or not the current script is being run in debug mode.
    """
    _shared_state = SharedDict()

    def __init__(self, mixed_precision: str=None, cpu: bool=False, dynamo_plugin=None, deepspeed_plugin=None, fsdp_plugin=None, megatron_lm_plugin=None, _from_accelerator: bool=False, **kwargs):
        self.__dict__ = self._shared_state
        if parse_flag_from_env('ACCELERATE_USE_CPU'):
            cpu = True
        if PartialState._shared_state == {}:
            PartialState(cpu, **kwargs)
        self.__dict__.update(PartialState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            self.deepspeed_plugin = None
            self.use_ipex = None
            mixed_precision = parse_choice_from_env('ACCELERATE_MIXED_PRECISION', 'no') if mixed_precision is None else mixed_precision.lower()
            if mixed_precision == 'fp8':
                if not is_fp8_available():
                    raise ValueError('Using `fp8` precision requires `transformer_engine` or `MS-AMP` to be installed.')
                elif not check_fp8_capability():
                    logger.warning(f'The current device has compute capability of {torch.cuda.get_device_capability()} which is insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace or higher, compute capability of 8.9 or higher). Will use FP16 instead.')
                    mixed_precision = 'fp16'
            self.dynamo_plugin = dynamo_plugin
            if not _from_accelerator:
                raise ValueError('Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` before using any functionality from the `accelerate` library.')
            self._mixed_precision = 'no' if self.distributed_type == DistributedType.DEEPSPEED else mixed_precision
            if self.distributed_type == DistributedType.XLA and is_torch_xla_available(check_is_tpu=True):
                if mixed_precision == 'bf16':
                    if os.environ.get('ACCELERATE_DOWNCAST_BF16'):
                        os.environ['XLA_USE_BF16'] = str(0)
                        os.environ['XLA_DOWNCAST_BF16'] = str(1)
                        self.downcast_bfloat = True
                    else:
                        os.environ['XLA_USE_BF16'] = str(1)
                        os.environ['XLA_DOWNCAST_BF16'] = str(0)
                        self.downcast_bfloat = False
            elif os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false') == 'true' and (not cpu):
                self.deepspeed_plugin = deepspeed_plugin
            elif self.distributed_type == DistributedType.MULTI_GPU:
                if os.environ.get('ACCELERATE_USE_FSDP', 'false') == 'true':
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != 'no':
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
                if os.environ.get('ACCELERATE_USE_MEGATRON_LM', 'false') == 'true':
                    self.distributed_type = DistributedType.MEGATRON_LM
                    megatron_lm_plugin.set_mixed_precision(self._mixed_precision)
                    self.megatron_lm_plugin = megatron_lm_plugin
            elif self.distributed_type == DistributedType.MULTI_NPU:
                if os.environ.get('ACCELERATE_USE_FSDP', 'false') == 'true':
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != 'no':
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
            elif self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
                if is_ipex_available():
                    'check if user disables it explicitly'
                    self.use_ipex = parse_flag_from_env('ACCELERATE_USE_IPEX', default=True)
                else:
                    self.use_ipex = False
                if self.distributed_type == DistributedType.MULTI_XPU:
                    if os.environ.get('ACCELERATE_USE_FSDP', 'false') == 'true':
                        self.distributed_type = DistributedType.FSDP
                        if self._mixed_precision != 'no':
                            fsdp_plugin.set_mixed_precision(self._mixed_precision)
                        self.fsdp_plugin = fsdp_plugin
            if self.dynamo_plugin.backend != DynamoBackend.NO and self._mixed_precision == 'no' and (self.device.type == 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
            PartialState._shared_state['distributed_type'] = self.distributed_type

    @property
    def initialized(self) -> bool:
        return self._shared_state != PartialState._shared_state

    def __repr__(self):
        repr = PartialState().__repr__() + f'\nMixed precision type: {self.mixed_precision}\n'
        if self.distributed_type == DistributedType.DEEPSPEED:
            repr += f'ds_config: {self.deepspeed_plugin.deepspeed_config}\n'
        return repr

    def _check_initialized(self, mixed_precision=None, cpu=None):
        """Checks if a modification is trying to be made and the `AcceleratorState` has already been initialized"""
        if self.initialized:
            err = 'AcceleratorState has already been initialized and cannot be changed, restart your runtime completely and pass `{flag}` to `Accelerator()`.'
            if cpu and self.device.type != 'cpu':
                raise ValueError(err.format(flag='cpu=True'))
            if mixed_precision is not None and mixed_precision != self._mixed_precision and (self.distributed_type != DistributedType.DEEPSPEED):
                raise ValueError(err.format(flag=f"mixed_precision='{mixed_precision}'"))

    @property
    def use_fp16(self):
        warnings.warn("The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use `AcceleratorState.mixed_precision == 'fp16'` instead.", FutureWarning)
        return self._mixed_precision != 'no'

    @property
    def mixed_precision(self):
        if self.distributed_type == DistributedType.DEEPSPEED:
            config = self.deepspeed_plugin.deepspeed_config
            if config.get('fp16', {}).get('enabled', False):
                mixed_precision = 'fp16'
            elif config.get('bf16', {}).get('enabled', False):
                mixed_precision = 'bf16'
            else:
                mixed_precision = 'no'
        else:
            mixed_precision = self._mixed_precision
        return mixed_precision

    @staticmethod
    def _reset_state(reset_partial_state: bool=False):
        """Resets `_shared_state`, is used internally and should not be called"""
        AcceleratorState._shared_state.clear()
        if reset_partial_state:
            PartialState._reset_state()

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return PartialState().use_distributed

    @property
    def is_last_process(self) -> bool:
        """Returns whether the current process is the last one"""
        return PartialState().is_last_process

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process"""
        return PartialState().is_main_process

    @property
    def is_local_main_process(self) -> bool:
        """Returns whether the current process is the main process on the local node"""
        return PartialState().is_local_main_process

    def wait_for_everyone(self):
        PartialState().wait_for_everyone()

    @contextmanager
    def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool=False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
                in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.


        Example:

        ```python
        # Assume there are two processes
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        with state.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        with PartialState().split_between_processes(inputs, apply_padding=apply_padding) as inputs:
            yield inputs

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().main_process_first():
            yield

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().local_main_process_first():
            yield

    def print(self, *args, **kwargs):
        PartialState().print(*args, **kwargs)