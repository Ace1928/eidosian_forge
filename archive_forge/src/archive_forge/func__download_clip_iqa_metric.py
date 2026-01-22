from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_iqa import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _download_clip_iqa_metric() -> None:
    _CLIPModel.from_pretrained('openai/clip-vit-large-patch14', resume_download=True)
    _CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', resume_download=True)