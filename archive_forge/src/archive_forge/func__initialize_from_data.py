import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
def _initialize_from_data(self, data: 'ImageDataType', mode: Optional[str]=None, file_type: Optional[str]=None) -> None:
    pil_image = util.get_module('PIL.Image', required='wandb.Image needs the PIL package. To get it, run "pip install pillow".')
    if util.is_matplotlib_typename(util.get_full_typename(data)):
        buf = BytesIO()
        util.ensure_matplotlib_figure(data).savefig(buf, format='png')
        self._image = pil_image.open(buf, formats=['PNG'])
    elif isinstance(data, pil_image.Image):
        self._image = data
    elif util.is_pytorch_tensor_typename(util.get_full_typename(data)):
        vis_util = util.get_module('torchvision.utils', 'torchvision is required to render images')
        if hasattr(data, 'requires_grad') and data.requires_grad:
            data = data.detach()
        if hasattr(data, 'dtype') and str(data.dtype) == 'torch.uint8':
            data = data.to(float)
        data = vis_util.make_grid(data, normalize=True)
        self._image = pil_image.fromarray(data.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    else:
        if hasattr(data, 'numpy'):
            data = data.numpy()
        if data.ndim > 2:
            data = data.squeeze()
        self._image = pil_image.fromarray(self.to_uint8(data), mode=mode or self.guess_mode(data))
    accepted_formats = ['png', 'jpg', 'jpeg', 'bmp']
    if file_type is None:
        self.format = 'png'
    else:
        self.format = file_type
    assert self.format in accepted_formats, f'file_type must be one of {accepted_formats}'
    tmp_path = os.path.join(MEDIA_TMP.name, runid.generate_id() + '.' + self.format)
    assert self._image is not None
    self._image.save(tmp_path, transparency=None)
    self._set_file(tmp_path, is_tmp=True)