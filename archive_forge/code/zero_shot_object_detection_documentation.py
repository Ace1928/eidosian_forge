from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import ChunkPipeline, build_pipeline_init_args

        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        