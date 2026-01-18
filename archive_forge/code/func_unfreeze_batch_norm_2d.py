from typing import Optional, Tuple, Union
import torch
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available, requires_backends
from ...utils.backbone_utils import BackboneMixin
from .configuration_timm_backbone import TimmBackboneConfig
def unfreeze_batch_norm_2d(self):
    timm.layers.unfreeze_batch_norm_2d(self._backbone)