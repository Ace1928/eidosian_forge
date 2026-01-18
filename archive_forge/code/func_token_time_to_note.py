import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def token_time_to_note(number, cutoff_time_idx, current_idx):
    current_idx += number
    if cutoff_time_idx is not None:
        current_idx = min(current_idx, cutoff_time_idx)
    return current_idx