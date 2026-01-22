import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _prepare_bias_variables(self, scores: torch.FloatTensor):
    vocabulary_size = scores.shape[-1]
    invalid_biases = []
    for sequence_ids in self.sequence_bias:
        for token_id in sequence_ids:
            if token_id >= vocabulary_size:
                invalid_biases.append(token_id)
    if len(invalid_biases) > 0:
        raise ValueError(f'The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: {invalid_biases}')
    self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
    for sequence_ids, bias in self.sequence_bias.items():
        if len(sequence_ids) == 1:
            self.length_1_bias[sequence_ids[-1]] = bias
    self.prepared_bias_variables = True