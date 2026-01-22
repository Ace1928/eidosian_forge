from typing import Dict, Literal, Sequence, Tuple, Union
from torch import Tensor
def _validate_iou_type_arg(iou_type: Union[Literal['bbox', 'segm'], Tuple[str]]='bbox') -> Tuple[str]:
    """Validate that iou type argument is correct."""
    allowed_iou_types = ('segm', 'bbox')
    if isinstance(iou_type, str):
        iou_type = (iou_type,)
    if any((tp not in allowed_iou_types for tp in iou_type)):
        raise ValueError(f'Expected argument `iou_type` to be one of {allowed_iou_types} or a list of, but got {iou_type}')
    return iou_type