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
class SGDWithPausingCompute(torch.optim.SGD):

    def __init__(self, *args, **kwargs) -> None:
        self.rank = kwargs['rank']
        del kwargs['rank']
        super().__init__(*args, **kwargs)

    def step(self, closure: Optional[Any]=None) -> Any:
        loss = super().step(closure=closure)
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(100000000)
            with torch.no_grad():
                for param_group in self.param_groups:
                    for param in param_group['params']:
                        param *= 1.0 + self.rank / 10.0
        return loss