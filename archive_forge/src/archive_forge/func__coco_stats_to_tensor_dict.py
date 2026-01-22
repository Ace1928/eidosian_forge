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
def _coco_stats_to_tensor_dict(self, stats: List[float], prefix: str) -> Dict[str, Tensor]:
    """Converts the output of COCOeval.stats to a dict of tensors."""
    mdt = self.max_detection_thresholds
    return {f'{prefix}map': torch.tensor([stats[0]], dtype=torch.float32), f'{prefix}map_50': torch.tensor([stats[1]], dtype=torch.float32), f'{prefix}map_75': torch.tensor([stats[2]], dtype=torch.float32), f'{prefix}map_small': torch.tensor([stats[3]], dtype=torch.float32), f'{prefix}map_medium': torch.tensor([stats[4]], dtype=torch.float32), f'{prefix}map_large': torch.tensor([stats[5]], dtype=torch.float32), f'{prefix}mar_{mdt[0]}': torch.tensor([stats[6]], dtype=torch.float32), f'{prefix}mar_{mdt[1]}': torch.tensor([stats[7]], dtype=torch.float32), f'{prefix}mar_{mdt[2]}': torch.tensor([stats[8]], dtype=torch.float32), f'{prefix}mar_small': torch.tensor([stats[9]], dtype=torch.float32), f'{prefix}mar_medium': torch.tensor([stats[10]], dtype=torch.float32), f'{prefix}mar_large': torch.tensor([stats[11]], dtype=torch.float32)}