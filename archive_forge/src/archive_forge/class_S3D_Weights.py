from functools import partial
from typing import Any, Callable, Optional
import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
class S3D_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(url='https://download.pytorch.org/models/s3d-d76dad2f.pth', transforms=partial(VideoClassification, crop_size=(224, 224), resize_size=(256, 256)), meta={'min_size': (224, 224), 'min_temporal_size': 14, 'categories': _KINETICS400_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/video_classification#s3d', '_docs': 'The weights aim to approximate the accuracy of the paper. The accuracies are estimated on clip-level with parameters `frame_rate=15`, `clips_per_video=1`, and `clip_len=128`.', 'num_params': 8320048, '_metrics': {'Kinetics-400': {'acc@1': 68.368, 'acc@5': 88.05}}, '_ops': 17.979, '_file_size': 31.972})
    DEFAULT = KINETICS400_V1