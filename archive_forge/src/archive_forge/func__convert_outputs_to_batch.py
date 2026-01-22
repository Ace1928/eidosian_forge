import collections
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from ray.train._internal.dl_predictor import TensorDtype
from ray.train.torch.torch_predictor import TorchPredictor
from ray.util.annotations import PublicAPI
def _convert_outputs_to_batch(outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    """Batch detection model outputs.

    TorchVision detection models return `List[Dict[Tensor]]`. Each `Dict` contain
    'boxes', 'labels, and 'scores'.

    This function batches values and returns a `Dict[str, List[Tensor]]`.
    """
    batch = collections.defaultdict(list)
    for output in outputs:
        for key, value in output.items():
            batch[key].append(value.cpu().detach())
    return batch