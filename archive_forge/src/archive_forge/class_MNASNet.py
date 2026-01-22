import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class MNASNet(torch.nn.Module):
    """MNASNet, as described in https://arxiv.org/abs/1807.11626. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha: float, num_classes: int=1000, dropout: float=0.2) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if alpha <= 0.0:
            raise ValueError(f'alpha should be greater than 0.0 instead of {alpha}')
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM), _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM), _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM), _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM), _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM), _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM), _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM), nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = x.mean([2, 3])
        return self.classifier(x)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, local_metadata: Dict, strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
        version = local_metadata.get('version', None)
        if version not in [1, 2]:
            raise ValueError(f'version shluld be set to 1 or 2 instead of {version}')
        if version == 1 and (not self.alpha == 1.0):
            depths = _get_depths(self.alpha)
            v1_stem = [nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(16, momentum=_BN_MOMENTUM), _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM)]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer
            self._version = 1
            warnings.warn('A new version of MNASNet model has been implemented. Your checkpoint was saved using the previous version. This checkpoint will load and work as before, but you may want to upgrade by training a newer model or transfer learning from an updated ImageNet checkpoint.', UserWarning)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)