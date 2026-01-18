from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args
def unnormalize(bbox):
    return self._get_bounding_box(torch.Tensor([width * bbox[0] / 1000, height * bbox[1] / 1000, width * bbox[2] / 1000, height * bbox[3] / 1000]))