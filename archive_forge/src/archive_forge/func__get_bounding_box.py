from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args
def _get_bounding_box(self, box: 'torch.Tensor') -> Dict[str, int]:
    """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
    if self.framework != 'pt':
        raise ValueError('The ObjectDetectionPipeline is only available in PyTorch.')
    xmin, ymin, xmax, ymax = box.int().tolist()
    bbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    return bbox