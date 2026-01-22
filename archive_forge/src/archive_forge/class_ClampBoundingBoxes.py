from typing import Any, Dict, Union
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
class ClampBoundingBoxes(Transform):
    """[BETA] Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.

    .. v2betastatus:: ClampBoundingBoxes transform

    """
    _transformed_types = (tv_tensors.BoundingBoxes,)

    def _transform(self, inpt: tv_tensors.BoundingBoxes, params: Dict[str, Any]) -> tv_tensors.BoundingBoxes:
        return F.clamp_bounding_boxes(inpt)