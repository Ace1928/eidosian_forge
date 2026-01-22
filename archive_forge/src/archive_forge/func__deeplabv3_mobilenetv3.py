from functools import partial
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead
def _deeplabv3_mobilenetv3(backbone: MobileNetV3, num_classes: int, aux: Optional[bool]) -> DeepLabV3:
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, '_is_cn', False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): 'out'}
    if aux:
        return_layers[str(aux_pos)] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)