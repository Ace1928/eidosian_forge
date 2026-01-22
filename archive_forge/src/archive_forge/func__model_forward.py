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
def _model_forward(training: bool, shape: torch.Size, dtype: torch.dtype) -> Optional[Tuple[SizeOrSizes, DtypeOrDtypes]]:
    try:
        if isinstance(shape, torch.Size):
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = tuple([torch.empty(s, dtype=d) for s, d in zip(shape, dtype)])
        model = PipeModel
        assert model.group
        set_device_based_on_group(model.group)
        model.train(training)
        result = model(tensor)
        if model.final_stage:
            global PipeResult
            PipeResult = result
            return (get_shapes(result), get_dtype(result))
        return None
    except Exception as e:
        print(f'_model_forward got {e}')
        raise e