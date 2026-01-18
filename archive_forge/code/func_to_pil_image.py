import math
import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from ..utils import _log_api_usage_once
from . import _functional_pil as F_pil, _functional_tensor as F_t
def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(to_pil_image)
    if not (isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError(f'pic should be Tensor or ndarray. Got {type(pic)}.')
    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError(f'pic should be 2/3 dimensional. Got {pic.ndimension()} dimensions.')
        elif pic.ndimension() == 2:
            pic = pic.unsqueeze(0)
        if pic.shape[-3] > 4:
            raise ValueError(f'pic should not have > 4 channels. Got {pic.shape[-3]} channels.')
    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError(f'pic should be 2/3 dimensional. Got {pic.ndim} dimensions.')
        elif pic.ndim == 2:
            pic = np.expand_dims(pic, 2)
        if pic.shape[-1] > 4:
            raise ValueError(f'pic should not have > 4 channels. Got {pic.shape[-1]} channels.')
    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))
    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, not {type(npimg)}')
    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(f'Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}')
        mode = expected_mode
    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f'Only modes {permitted_2_channel_modes} are supported for 2D inputs')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'
    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f'Only modes {permitted_4_channel_modes} are supported for 4D inputs')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f'Only modes {permitted_3_channel_modes} are supported for 3D inputs')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
    if mode is None:
        raise TypeError(f'Input type {npimg.dtype} is not supported')
    return Image.fromarray(npimg, mode=mode)