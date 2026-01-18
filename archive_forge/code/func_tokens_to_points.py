import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def tokens_to_points(tokens, original_size):
    while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (None, None):
        start, end = pair
        if end != start + 3:
            continue
        coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1:end])
        scale = scale_factor_to_fit(original_size)
        x, y = [2 * int(float(c) / scale) for c in coords]
        replacement = f' {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}'
        replacement = self.tokenizer.tokenize(replacement)[1:]
        replacement = self.tokenizer.convert_tokens_to_ids(replacement)
        replacement = torch.tensor(replacement).to(tokens)
        tokens = torch.cat([tokens[:start], replacement, tokens[end + 1:]], 0)
    return tokens