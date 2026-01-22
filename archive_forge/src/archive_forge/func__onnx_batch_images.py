import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
@torch.jit.unused
def _onnx_batch_images(self, images: List[Tensor], size_divisible: int=32) -> Tensor:
    max_size = []
    for i in range(images[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    stride = size_divisible
    max_size[1] = (torch.ceil(max_size[1].to(torch.float32) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil(max_size[2].to(torch.float32) / stride) * stride).to(torch.int64)
    max_size = tuple(max_size)
    padded_imgs = []
    for img in images:
        padding = [s1 - s2 for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
    return torch.stack(padded_imgs)