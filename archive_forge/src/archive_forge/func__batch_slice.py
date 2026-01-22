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
def _batch_slice(self, slice_: slice) -> 'SampleBatch':
    """Helper method to handle SampleBatch slicing using a slice object.

        The returned SampleBatch uses the same underlying data object as
        `self`, so changing the slice will also change `self`.

        Note that only zero or positive bounds are allowed for both start
        and stop values. The slice step must be 1 (or None, which is the
        same).

        Args:
            slice_: The python slice object to slice by.

        Returns:
            A new SampleBatch, however "linking" into the same data
            (sliced) as self.
        """
    start = slice_.start or 0
    stop = slice_.stop or len(self[SampleBatch.SEQ_LENS])
    if stop > len(self):
        stop = len(self)
    assert start >= 0 and stop >= 0 and (slice_.step in [1, None])
    data = tree.map_structure(lambda value: value[start:stop], self)
    return SampleBatch(data, _is_training=self.is_training, _time_major=self.time_major, _num_grad_updates=self.num_grad_updates)