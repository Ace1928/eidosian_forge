import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def tokens_to_boxes(tokens, original_size):
    while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (None, None):
        start, end = pair
        if end != start + 5:
            continue
        coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
        scale = scale_factor_to_fit(original_size)
        top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]
        replacement = f' {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}'
        replacement = self.tokenizer.tokenize(replacement)[1:]
        replacement = self.tokenizer.convert_tokens_to_ids(replacement)
        replacement = torch.tensor(replacement).to(tokens)
        tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
    return tokens