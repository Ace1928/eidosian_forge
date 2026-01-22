import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
class PushState(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        if len(args) == 1:
            return args[0]
        return args

    @staticmethod
    def backward(ctx, *grad_outs):
        self._enter_module(name)
        return grad_outs