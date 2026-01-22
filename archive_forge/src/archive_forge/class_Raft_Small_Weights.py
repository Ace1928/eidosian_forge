from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class Raft_Small_Weights(WeightsEnum):
    """The metrics reported here are as follows.

    ``epe`` is the "end-point-error" and indicates how far (in pixels) the
    predicted flow is from its true value. This is averaged over all pixels
    of all images. ``per_image_epe`` is similar, but the average is different:
    the epe is first computed on each image independently, and then averaged
    over all images. This corresponds to "Fl-epe" (sometimes written "F1-epe")
    in the original paper, and it's only used on Kitti. ``fl-all`` is also a
    Kitti-specific metric, defined by the author of the dataset and used for the
    Kitti leaderboard. It corresponds to the average of pixels whose epe is
    either <3px, or <5% of flow's 2-norm.
    """
    C_T_V1 = Weights(url='https://download.pytorch.org/models/raft_small_C_T_V1-ad48884c.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 990162, 'recipe': 'https://github.com/princeton-vl/RAFT', '_metrics': {'Sintel-Train-Cleanpass': {'epe': 2.1231}, 'Sintel-Train-Finalpass': {'epe': 3.279}, 'Kitti-Train': {'per_image_epe': 7.6557, 'fl_all': 25.2801}}, '_ops': 47.655, '_file_size': 3.821, '_docs': 'These weights were ported from the original paper. They\n            are trained on :class:`~torchvision.datasets.FlyingChairs` +\n            :class:`~torchvision.datasets.FlyingThings3D`.'})
    C_T_V2 = Weights(url='https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 990162, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/optical_flow', '_metrics': {'Sintel-Train-Cleanpass': {'epe': 1.9901}, 'Sintel-Train-Finalpass': {'epe': 3.2831}, 'Kitti-Train': {'per_image_epe': 7.5978, 'fl_all': 25.2369}}, '_ops': 47.655, '_file_size': 3.821, '_docs': 'These weights were trained from scratch on\n            :class:`~torchvision.datasets.FlyingChairs` +\n            :class:`~torchvision.datasets.FlyingThings3D`.'})
    DEFAULT = C_T_V2