import collections
import itertools
import json
import os
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, logging
def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in self.model_input_names
    required_input = encoded_inputs[self.model_input_names[0]]
    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)
    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
    if return_attention_mask and 'attention_mask' not in encoded_inputs:
        encoded_inputs['attention_mask'] = [1] * len(required_input)
    if needs_to_be_padded:
        difference = max_length - len(required_input)
        if self.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'] + [0] * difference
            if 'token_type_ids' in encoded_inputs:
                encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'] + [self.pad_token_type_id] * difference
            if 'special_tokens_mask' in encoded_inputs:
                encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
            for key in ['input_shape_ids', 'input_pronunciation_ids']:
                if key in encoded_inputs:
                    encoded_inputs[key] = encoded_inputs[key] + [self.pad_token_id] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
        elif self.padding_side == 'left':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = [0] * difference + encoded_inputs['attention_mask']
            if 'token_type_ids' in encoded_inputs:
                encoded_inputs['token_type_ids'] = [self.pad_token_type_id] * difference + encoded_inputs['token_type_ids']
            if 'special_tokens_mask' in encoded_inputs:
                encoded_inputs['special_tokens_mask'] = [1] * difference + encoded_inputs['special_tokens_mask']
            for key in ['input_shape_ids', 'input_pronunciation_ids']:
                if key in encoded_inputs:
                    encoded_inputs[key] = [self.pad_token_id] * difference + encoded_inputs[key]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
        else:
            raise ValueError('Invalid padding strategy:' + str(self.padding_side))
    return encoded_inputs