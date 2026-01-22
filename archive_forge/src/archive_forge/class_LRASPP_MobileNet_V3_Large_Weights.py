from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(url='https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth', transforms=partial(SemanticSegmentation, resize_size=520), meta={'num_params': 3221538, 'categories': _VOC_CATEGORIES, 'min_size': (1, 1), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/segmentation#lraspp_mobilenet_v3_large', '_metrics': {'COCO-val2017-VOC-labels': {'miou': 57.9, 'pixel_acc': 91.2}}, '_ops': 2.086, '_file_size': 12.49, '_docs': '\n                These weights were trained on a subset of COCO, using only the 20 categories that are present in the\n                Pascal VOC dataset.\n            '})
    DEFAULT = COCO_WITH_VOC_LABELS_V1