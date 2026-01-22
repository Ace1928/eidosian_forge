import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
@dataclass
class FlavaLosses(ModelOutput):
    """Class representing pretraining losses from FLAVA model

    Args:
        mim (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels` and `pixel_values` are present, `input_ids_masked` is absent and `mim_weight` > 0.:
            Masked Image Modeling loss as used in BeIT calculated only for unimodal image data.
        mlm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels` and `input_ids_masked` are present, `pixel_values` is absent and `mlm_weight` > 0.:
            Masked Language Modeling loss as used in BERT calculated only for unimodal text data.
        itm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `itm_labels`, `input_ids_masked`, `pixel_values` are present and `itm_weight` > 0.:
            Image Text Matching (ITM) loss calculated for paired image-text data. Note that ITM loss is calculated on
            masked pairs in FLAVA.
        global_contrastive (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `input_ids` and `pixel_values` are present and `global_contrastive_weight` > 0.:
            Contrastive loss for image-text similarity similar to CLIP but calculated globally for paired image-text
            data. This is calculated on unmasked images and texts.
        mmm_image (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_image_weight` > 0.:
            Masked Multimodal Modeling loss's image component calculated on paired image-text data.
        mmm_text (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_text_weight` > 0.:
            Masked Multimodal Modeling loss's text component calculated on paired image-text data.
    """
    mim: Optional[torch.FloatTensor] = None
    mlm: Optional[torch.FloatTensor] = None
    itm: Optional[torch.FloatTensor] = None
    global_contrastive: Optional[torch.FloatTensor] = None
    mmm_image: Optional[torch.FloatTensor] = None
    mmm_text: Optional[torch.FloatTensor] = None

    def all_none(self) -> bool:
        all_none = True
        for v in self.values():
            if v is not None:
                all_none = False
                break
        return all_none