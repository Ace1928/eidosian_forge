import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if not len(trunc_tokens) >= 1:
            raise ValueError('Sequence length to be truncated must be no less than one')
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()