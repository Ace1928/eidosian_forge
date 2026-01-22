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
class CUDAWarmupNode:
    """
    Simplified Wrapper around A CUDA Model that wraps outputs in storage refs and exposes
    apis to get the live storages in the current chain of warmup.

    A CUDAWarmupNode may have either CUDAGraphNode or CUDAWarmupNode as a parent, but may only have
    CUDAWarmupNode as children, because we cannot record or execute with tensors which do not have stable
    memory addresses.

    CUDAWarmupNode and CUDAGraphNode have a number of differences that make it easier to use separate classes.
    - Much of the CUDAGraphNode logic & initialization is based on the tensor properties of first recording. In the
    first instance of warmup, these are not finalized yet.
    - All Inputs to the RecordedFunction must be copied over to the cuda graph memory pool, this is unnecessary in warmup.
    - CUDAWarmup is only used once and so does not need to optimize as much bookkeeping. It is much simpler.

    NB: this class and CUDAGraphNode need to expose `path_live_weakrefs`, `all_outputs_are_dead`, and
    `self.outputs_weakrefs`, `stack_traces`, and `tensor_weakrefs` for compatibility.
    """

    def __init__(self, wrapped_function: WrappedFunction, parent, cuda_graphs_pool: Tuple[int, int], existing_cuda_graph: Optional[torch.cuda.CUDAGraph], device_index: int, stack_traces: Optional[StackTraces], stream: torch.cuda.Stream, already_warm: bool):
        self.wrapped_function = wrapped_function
        self.parent = parent
        self.cuda_graphs_pool = cuda_graphs_pool
        self.outputs_weakrefs: List[Optional[StorageWeakRefWrapper]] = []
        self.tensor_weakrefs: List[Optional[TensorWeakRef]] = []
        self.existing_cuda_graph = existing_cuda_graph
        self.has_run = False
        self.device_index = device_index
        self.stack_traces = stack_traces
        self.stream = stream
        self.already_warm = already_warm

    def run(self, new_inputs):
        assert not self.has_run, 'Wrapped function should never be run twice'
        existing_path_data_ptrs = {t.data_ptr() for t in self.path_live_weakrefs() if t()}

        def get_non_cudagraph_inps():
            non_cudagraph_inps = set()
            for t in itertools.chain(new_inputs, self.wrapped_function.constants):
                if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in existing_path_data_ptrs:
                    non_cudagraph_inps.add(t.untyped_storage().data_ptr())
            return non_cudagraph_inps
        non_cudagraph_inps = get_non_cudagraph_inps()
        if config.triton.slow_path_cudagraph_asserts and (not self.already_warm):
            refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)
        with torch.cuda.device(self.device_index), disable_conv_cache_emptying(), clear_cublas_manager(), _use_cuda_memory_pool_manager(self.device_index, self.cuda_graphs_pool, self.stream), get_history_recording():
            out = self.wrapped_function.model(new_inputs)
        torch.cuda.synchronize()
        assert len(new_inputs) == 0

        def add_ref(o):
            return o is not None and isinstance(o, torch.Tensor) and o.is_cuda and (o.untyped_storage().data_ptr() not in non_cudagraph_inps) and (o.untyped_storage().data_ptr() != 0)
        self.outputs_weakrefs.extend([map_to_ref(o) if add_ref(o) else None for o in out])
        self.tensor_weakrefs.extend([TensorWeakRef(o) if add_ref(o) else None for o in out])
        if config.triton.slow_path_cudagraph_asserts and (not self.already_warm):
            out_refs = self.path_live_weakrefs()
            new_storages = [t for t in out_refs if t.data_ptr() not in non_cudagraph_inps]
            check_memory_pool(self.device_index, self.cuda_graphs_pool, new_storages)
        return out

    @property
    def _path_from_root(self):
        nodes = []
        node = self
        while node:
            nodes.append(node)
            node = node.parent
        yield from reversed(nodes)

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        """Returns all live storages weakrefs that created by nodes in this path"""
        for node in self._path_from_root:
            for output in node.outputs_weakrefs:
                if is_live(output):
                    yield output

    def all_outputs_are_dead(self):
        return not list(self.path_live_weakrefs())