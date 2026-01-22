import warnings
from typing import Callable, Dict, List, Optional, Union
from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from .. import mobilenet, resnet
from .._api import _get_enum_from_fn, WeightsEnum
from .._utils import handle_legacy_interface, IntermediateLayerGetter
def _validate_trainable_layers(is_trained: bool, trainable_backbone_layers: Optional[int], max_value: int, default_value: int) -> int:
    if not is_trained:
        if trainable_backbone_layers is not None:
            warnings.warn(f'Changing trainable_backbone_layers has no effect if neither pretrained nor pretrained_backbone have been set to True, falling back to trainable_backbone_layers={max_value} so that all layers are trainable')
        trainable_backbone_layers = max_value
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    if trainable_backbone_layers < 0 or trainable_backbone_layers > max_value:
        raise ValueError(f'Trainable backbone layers should be in the range [0,{max_value}], got {trainable_backbone_layers} ')
    return trainable_backbone_layers