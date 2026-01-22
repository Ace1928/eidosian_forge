import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
class _CheckpointFrame:

    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn):
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[ReferenceType] = []
        self.recomputed: DefaultDict[int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]] = defaultdict(weakref.WeakKeyDictionary)
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)
        self.early_stop = early_stop
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

    def check_recomputed_tensors_match(self, gid):
        if self.ignore_saved_mismatch:
            return
        if not len(self.weak_holders) == self.recomp_counter[gid]:
            raise CheckpointError(f'torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.\nNumber of tensors saved during forward: {len(self.weak_holders)}\nNumber of tensors saved during recomputation: {self.recomp_counter[gid]}')
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            _internal_assert(gid in holder.handles)
            _internal_assert(holder.handles[gid] is not None)
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))
        if len(nb_meta_different) > 0:
            mismatched_tensors = ''
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += f'tensor at position {idx}:\nsaved metadata: {x_meta}\nrecomputed metadata: {recomputed_meta}\n'
            raise CheckpointError(f'torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.\n{mismatched_tensors}')