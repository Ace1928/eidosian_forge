import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
def torch_spawn(world_sizes: Optional[List[int]]=None) -> Callable:
    if world_sizes is None:
        world_sizes = get_world_sizes()

    def prepare_test(func: Callable) -> Callable:
        """Function called with the test function as the argument. Generates a
        replacement which serves as the actual test function."""
        name = func.__name__
        parameters = inspect.signature(func).parameters
        if name.startswith('test'):
            raise ValueError(f"Tests marked with @torch_spawn (i.e. '{name}') should not have names beginning in 'test' as they will be picked up by pytest without running the spawn wrapper")

        @functools.wraps(func)
        def replacement(*args: Any, **kwargs: Any) -> None:
            assert args == tuple()
            assert world_sizes is not None
            args = tuple((kwargs[p] for p in parameters if p != 'rank'))
            error_queue = multiprocessing.get_context('spawn').SimpleQueue()
            if 'OMPI_COMM_WORLD_RANK' in os.environ:
                global filename_mpi
                if filename_mpi is None:
                    filename_mpi = tempfile.mkstemp()[1]
                os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
                os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
                torch.distributed.init_process_group('mpi', init_method=f'file://{filename_mpi}')
                world_size = torch.distributed.get_world_size()
                destroy_model_parallel()
                initialize_model_parallel(1, world_size)
                torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
                if world_size in world_sizes:
                    try:
                        func(*args)
                        teardown()
                    except BaseException as e:
                        teardown()
                        import traceback
                        print(f'{traceback.format_exc()}')
                        raise e
                else:
                    pytest.skip("Requested world size doesn't match current world size")
            else:
                spawn_for_all_world_sizes(worker_process, world_sizes, (func, args, error_queue))
            if not error_queue.empty():
                msg = error_queue.get()
                pytest.skip(msg)
        current_frame = inspect.currentframe()
        assert current_frame is not None
        caller_module = inspect.getmodule(current_frame.f_back)
        setattr(caller_module, f'test_{name}', replacement)
        return func
    return prepare_test