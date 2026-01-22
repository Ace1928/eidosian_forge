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
def _prepare_backend(self, cpu: bool=False, sagemaker_dp=False, backend: str=None) -> tuple[str, DistributedType]:
    """Prepares any imports needed before initializing the distributed backend and sets `self.backend` properly"""
    distributed_type = None
    if sagemaker_dp:
        import smdistributed.dataparallel.torch.torch_smddp
        backend = 'smddp'
        distributed_type = DistributedType.MULTI_GPU
    elif is_torch_xla_available():
        backend = 'xla'
        distributed_type = DistributedType.XLA
    elif int(os.environ.get('LOCAL_RANK', -1)) != -1 and (not cpu):
        if is_mlu_available():
            backend = 'cncl'
            distributed_type = DistributedType.MULTI_MLU
        elif torch.cuda.is_available():
            if backend is None:
                backend = 'nccl'
            distributed_type = DistributedType.MULTI_GPU
        elif is_npu_available():
            backend = 'hccl'
            distributed_type = DistributedType.MULTI_NPU
    if distributed_type is None and (int(os.environ.get('LOCAL_RANK', -1)) != -1 or get_int_from_env(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1) > 1):
        if not cpu and is_xpu_available():
            distributed_type = DistributedType.MULTI_XPU
        else:
            distributed_type = DistributedType.MULTI_CPU
        if backend in (None, 'ccl') and is_ccl_available() and (get_int_from_env(['CCL_WORKER_COUNT'], 0) > 0 or distributed_type == DistributedType.MULTI_XPU):
            if get_ccl_version() >= '1.12':
                import oneccl_bindings_for_pytorch
            else:
                import torch_ccl
            backend = 'ccl'
        elif backend in (None, 'mpi') and torch.distributed.is_mpi_available():
            backend = 'mpi'
        else:
            backend = 'gloo'
    if distributed_type is None:
        distributed_type = DistributedType.NO
    return (backend, distributed_type)