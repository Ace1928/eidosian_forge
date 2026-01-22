from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
@staticmethod
def _recv_result(model: AsyncPipe, shapes: SizeOrSizes, dtypes: DtypeOrDtypes, message: PipeMessage) -> TensorOrTensors:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    assert model.pipeline
    transport = model.pipeline.transport
    if isinstance(shapes, torch.Size):
        message.tensor_shapes = [cast(torch.Size, shapes)]
        message.tensor_dtypes = [cast(torch.dtype, dtypes)]
        message = transport.recv_message_tensors(message)
        return message.tensors[0]
    else:
        message.tensor_shapes = cast(List[torch.Size], shapes)
        message.tensor_dtypes = cast(List[torch.dtype], dtypes)
        message = transport.recv_message_tensors(message)
        return message.tensors