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
class Raft_Large_Weights(WeightsEnum):
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
    C_T_V1 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_V1-22a6c225.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/princeton-vl/RAFT', '_metrics': {'Sintel-Train-Cleanpass': {'epe': 1.4411}, 'Sintel-Train-Finalpass': {'epe': 2.7894}, 'Kitti-Train': {'per_image_epe': 5.0172, 'fl_all': 17.4506}}, '_ops': 211.007, '_file_size': 20.129, '_docs': 'These weights were ported from the original paper. They\n            are trained on :class:`~torchvision.datasets.FlyingChairs` +\n            :class:`~torchvision.datasets.FlyingThings3D`.'})
    C_T_V2 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_V2-1bb1363a.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/optical_flow', '_metrics': {'Sintel-Train-Cleanpass': {'epe': 1.3822}, 'Sintel-Train-Finalpass': {'epe': 2.7161}, 'Kitti-Train': {'per_image_epe': 4.5118, 'fl_all': 16.0679}}, '_ops': 211.007, '_file_size': 20.129, '_docs': 'These weights were trained from scratch on\n            :class:`~torchvision.datasets.FlyingChairs` +\n            :class:`~torchvision.datasets.FlyingThings3D`.'})
    C_T_SKHT_V1 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_SKHT_V1-0b8c9e55.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/princeton-vl/RAFT', '_metrics': {'Sintel-Test-Cleanpass': {'epe': 1.94}, 'Sintel-Test-Finalpass': {'epe': 3.18}}, '_ops': 211.007, '_file_size': 20.129, '_docs': '\n                These weights were ported from the original paper. They are\n                trained on :class:`~torchvision.datasets.FlyingChairs` +\n                :class:`~torchvision.datasets.FlyingThings3D` and fine-tuned on\n                Sintel. The Sintel fine-tuning step is a combination of\n                :class:`~torchvision.datasets.Sintel`,\n                :class:`~torchvision.datasets.KittiFlow`,\n                :class:`~torchvision.datasets.HD1K`, and\n                :class:`~torchvision.datasets.FlyingThings3D` (clean pass).\n            '})
    C_T_SKHT_V2 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/optical_flow', '_metrics': {'Sintel-Test-Cleanpass': {'epe': 1.819}, 'Sintel-Test-Finalpass': {'epe': 3.067}}, '_ops': 211.007, '_file_size': 20.129, '_docs': '\n                These weights were trained from scratch. They are\n                pre-trained on :class:`~torchvision.datasets.FlyingChairs` +\n                :class:`~torchvision.datasets.FlyingThings3D` and then\n                fine-tuned on Sintel. The Sintel fine-tuning step is a\n                combination of :class:`~torchvision.datasets.Sintel`,\n                :class:`~torchvision.datasets.KittiFlow`,\n                :class:`~torchvision.datasets.HD1K`, and\n                :class:`~torchvision.datasets.FlyingThings3D` (clean pass).\n            '})
    C_T_SKHT_K_V1 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_SKHT_K_V1-4a6a5039.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/princeton-vl/RAFT', '_metrics': {'Kitti-Test': {'fl_all': 5.1}}, '_ops': 211.007, '_file_size': 20.129, '_docs': '\n                These weights were ported from the original paper. They are\n                pre-trained on :class:`~torchvision.datasets.FlyingChairs` +\n                :class:`~torchvision.datasets.FlyingThings3D`,\n                fine-tuned on Sintel, and then fine-tuned on\n                :class:`~torchvision.datasets.KittiFlow`. The Sintel fine-tuning\n                step was described above.\n            '})
    C_T_SKHT_K_V2 = Weights(url='https://download.pytorch.org/models/raft_large_C_T_SKHT_K_V2-b5c70766.pth', transforms=OpticalFlow, meta={**_COMMON_META, 'num_params': 5257536, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/optical_flow', '_metrics': {'Kitti-Test': {'fl_all': 5.19}}, '_ops': 211.007, '_file_size': 20.129, '_docs': '\n                These weights were trained from scratch. They are\n                pre-trained on :class:`~torchvision.datasets.FlyingChairs` +\n                :class:`~torchvision.datasets.FlyingThings3D`,\n                fine-tuned on Sintel, and then fine-tuned on\n                :class:`~torchvision.datasets.KittiFlow`. The Sintel fine-tuning\n                step was described above.\n            '})
    DEFAULT = C_T_SKHT_V2