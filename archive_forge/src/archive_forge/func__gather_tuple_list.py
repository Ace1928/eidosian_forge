import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
@staticmethod
def _gather_tuple_list(list_to_gather: List[Tuple], process_group: Optional[Any]=None) -> List[Any]:
    """Gather a list of tuples over multiple devices.

        Args:
            list_to_gather: input list of tuples that should be gathered across devices
            process_group: process group to gather the list of tuples

        Returns:
            list of tuples gathered across devices

        """
    world_size = dist.get_world_size(group=process_group)
    dist.barrier(group=process_group)
    list_gathered = [None for _ in range(world_size)]
    dist.all_gather_object(list_gathered, list_to_gather, group=process_group)
    return [list_gathered[rank][idx] for idx in range(len(list_gathered[0])) for rank in range(world_size)]