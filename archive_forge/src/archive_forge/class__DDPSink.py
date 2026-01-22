import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type
import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch._utils import _get_device_index
from ..modules import Module
from .scatter_gather import gather, scatter_kwargs  # noqa: F401
class _DDPSink(Function):

    @staticmethod
    def forward(ctx, ddp_weakref, *inputs):
        ctx.set_materialize_grads(False)
        ctx.ddp_weakref = ddp_weakref
        ret = tuple((inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs))
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        ddp_weakref = ctx.ddp_weakref()
        reducer = ddp_weakref.reducer
        static_graph = ddp_weakref.static_graph
        delay_ar_enqueued = static_graph and ddp_weakref._static_graph_delay_allreduce_enqueued
        if static_graph and (not delay_ar_enqueued):
            Variable._execution_engine.queue_callback(reducer._delay_all_reduce)
            ddp_weakref._static_graph_delay_allreduce_enqueued = True
        return (None, *grad_outputs)