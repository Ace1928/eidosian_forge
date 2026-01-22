from enum import Enum
from functools import reduce
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from parlai.agents.transformer.modules import (
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
def _forward_late_fusion(self, src_tokens: Optional[torch.LongTensor], image_features: Optional[Union[List[object], torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
        Encode images with context.

        Encodes tokens (if given) and images (if given) separately. Combines via either
        addition, prepending, or appending the image embedding to the context embedding.
        """
    context_encoded = context_mask = None
    image_encoded = extra_masks = None
    if src_tokens is not None and image_features is not None:
        assert src_tokens.size(0) == len(image_features)
    if src_tokens is not None:
        context_encoded, context_mask = super().forward(src_tokens)
    if image_features is not None:
        image_encoded, extra_masks = self.encode_images(image_features)
    if all((enc is None for enc in [context_encoded, image_encoded])):
        raise RuntimeError('You are providing Image+Seq2Seq with no input.\nIf you are using a text-based task, make sure the first turn has text (e.g. a __SILENCE__ token if the model starts the convo).\nIf you are using an image-based task, make sure --image-mode is set correctly.')
    if self.image_combination_mode == 'add':
        full_enc = self._add([context_encoded, image_encoded])
        full_mask = context_mask
    elif self.image_combination_mode == 'append':
        full_enc = self._cat([context_encoded, image_encoded])
        full_mask = self._cat([context_mask, extra_masks])
    elif self.image_combination_mode == 'prepend':
        full_enc = self._cat([image_encoded, context_encoded])
        full_mask = self._cat([extra_masks, context_mask])
    else:
        raise ValueError('Image combination mode not recognized!')
    if full_enc.dtype == torch.half:
        full_enc, full_mask = self._fix_for_fp16(full_enc=full_enc, full_mask=full_mask)
    return (full_enc, full_mask)