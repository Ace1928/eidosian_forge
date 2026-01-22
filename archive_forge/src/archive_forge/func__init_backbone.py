import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def _init_backbone(self, config) -> None:
    """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
    self.config = config
    self.use_timm_backbone = getattr(config, 'use_timm_backbone', False)
    self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS
    if self.backbone_type == BackboneType.TIMM:
        self._init_timm_backbone(config)
    elif self.backbone_type == BackboneType.TRANSFORMERS:
        self._init_transformers_backbone(config)
    else:
        raise ValueError(f'backbone_type {self.backbone_type} not supported.')