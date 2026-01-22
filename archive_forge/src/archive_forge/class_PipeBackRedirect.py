from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
class PipeBackRedirect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, dest, event, message, transport, futures):
        ctx.dest = dest
        ctx.event = event
        ctx.message = message
        ctx.transport = transport
        ctx.futures = futures
        return inputs

    @staticmethod
    def backward(ctx, *grad):
        ctx.message.tensors = tuple(grad)
        ctx.transport.send_message(ctx.message, sync=False, skip_header=True)
        ctx.event.set()
        return (None, None, None, None, None, None)