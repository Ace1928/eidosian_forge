from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
def upsample_flow(flow, up_mask: Optional[Tensor]=None, factor: int=8):
    """Upsample flow by the input factor (default 8).

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights. See paper page 8 and appendix B.
    Note that in appendix B the picture assumes a downsample factor of 4 instead of 8.
    """
    batch_size, num_channels, h, w = flow.shape
    new_h, new_w = (h * factor, w * factor)
    if up_mask is None:
        return factor * F.interpolate(flow, size=(new_h, new_w), mode='bilinear', align_corners=True)
    up_mask = up_mask.view(batch_size, 1, 9, factor, factor, h, w)
    up_mask = torch.softmax(up_mask, dim=2)
    upsampled_flow = F.unfold(factor * flow, kernel_size=3, padding=1).view(batch_size, num_channels, 9, 1, 1, h, w)
    upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)
    return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, num_channels, new_h, new_w)