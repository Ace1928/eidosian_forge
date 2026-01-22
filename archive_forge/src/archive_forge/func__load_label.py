import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _load_label(batch, targets, major_axis):
    """Load label into sliced arrays."""
    if isinstance(batch, list):
        new_batch = []
        for i in range(len(targets)):
            new_batch.append([b.label[i] for b in batch])
        new_targets = [[dst for _, dst in d_target] for d_target in targets]
        _load_general(new_batch, new_targets, major_axis)
    else:
        _load_general(batch.label, targets, major_axis)