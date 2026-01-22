important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
def _initialize_cached_tensors(self):
    assert len(self.outputs_weakrefs) == len(self.outputs_metadata)
    for i, (storage_info, metadata, make_cached) in enumerate(zip(self.output_storage_alias, self.outputs_metadata, self.unaliased_in_all_paths)):
        if not make_cached:
            self.cached_tensor_outputs.append(None)
            continue
        assert storage_info is UnaliasedStorage
        assert isinstance(metadata, dict)
        s = self.create_storage(metadata)
        out = self._reconstruct_from_tensor_metadata(metadata, storage=s)
        torch._C._add_cached_tensor(out)
        self_ref = weakref.ref(self)

        def check_refcount(i):
            self_loc = self_ref()
            if self_loc is None:
                return False
            return self_loc.get_output_refcount(i) == 2
        check = functools.partial(check_refcount, i=i)
        self.outputs_weakrefs[i] = StorageWeakRefWrapper(out, extra_ref_check=check)
        self.cached_tensor_outputs.append(out)