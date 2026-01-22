import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@Deprecated(error=False)
def _get_slice_indices(self, slice_size):
    data_slices = []
    data_slices_states = []
    if self.get(SampleBatch.SEQ_LENS) is not None and len(self[SampleBatch.SEQ_LENS]) > 0:
        assert np.all(self[SampleBatch.SEQ_LENS] < slice_size), 'ERROR: `slice_size` must be larger than the max. seq-len in the batch!'
        start_pos = 0
        current_slize_size = 0
        actual_slice_idx = 0
        start_idx = 0
        idx = 0
        while idx < len(self[SampleBatch.SEQ_LENS]):
            seq_len = self[SampleBatch.SEQ_LENS][idx]
            current_slize_size += seq_len
            actual_slice_idx += seq_len if not self.zero_padded else self.max_seq_len
            if current_slize_size >= slice_size:
                end_idx = idx + 1
                if not self.zero_padded:
                    data_slices.append((start_pos, start_pos + slice_size))
                    start_pos += slice_size
                    if current_slize_size > slice_size:
                        overhead = current_slize_size - slice_size
                        start_pos -= seq_len - overhead
                        idx -= 1
                else:
                    data_slices.append((start_pos, actual_slice_idx))
                    start_pos = actual_slice_idx
                data_slices_states.append((start_idx, end_idx))
                current_slize_size = 0
                start_idx = idx + 1
            idx += 1
    else:
        i = 0
        while i < self.count:
            data_slices.append((i, i + slice_size))
            i += slice_size
    return (data_slices, data_slices_states)