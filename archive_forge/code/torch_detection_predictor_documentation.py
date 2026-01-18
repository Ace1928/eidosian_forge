import collections
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from ray.train._internal.dl_predictor import TensorDtype
from ray.train.torch.torch_predictor import TorchPredictor
from ray.util.annotations import PublicAPI
Batch detection model outputs.

    TorchVision detection models return `List[Dict[Tensor]]`. Each `Dict` contain
    'boxes', 'labels, and 'scores'.

    This function batches values and returns a `Dict[str, List[Tensor]]`.
    