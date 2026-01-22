import torch
from torch import Tensor
def _box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    x, y, w, h = boxes.unbind(-1)
    boxes = torch.stack([x, y, x + w, y + h], dim=-1)
    return boxes