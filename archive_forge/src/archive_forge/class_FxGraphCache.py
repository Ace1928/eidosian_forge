from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import (
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.hub import _Faketqdm, tqdm
import torch
from ctypes import cdll
class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), 'fxgraph')

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def _filter_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the SymInt objects from the input list.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt)]

    @staticmethod
    def _get_shape_env() -> ShapeEnv:
        """
        Helper to get the shape env from the tracing context.
        """
        return torch._guards.TracingContext.get().fake_mode.shape_env

    @staticmethod
    def _lookup_graph(key: str, example_inputs: List[torch.Tensor]) -> Optional[CompiledFxGraph]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        """
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            return None
        for path in sorted(os.listdir(subdir)):
            with open(os.path.join(subdir, path), 'rb') as f:
                graph: CompiledFxGraph = pickle.load(f)
            guards_expr = graph.guards_expr
            if not guards_expr:
                return graph
            shape_env = FxGraphCache._get_shape_env()
            symints = FxGraphCache._filter_symints(example_inputs)
            assert all((has_hint(s) for s in symints))
            hints = [hint_int(s) for s in symints]
            hit = bool(shape_env.evaluate_guards_expression(guards_expr, hints))
            log.debug('fx graph cache key %s evaluating guards for %s with values %s => %s', key, guards_expr, hints, hit)
            if hit:
                check = bool(shape_env.evaluate_guards_expression(guards_expr, symints))
                assert check is True
                log.debug('fx graph cache key %s post-load guards: %s', key, shape_env.guards)
                return graph
        return None

    @staticmethod
    def _save_graph(key: str, compiled_graph: CompiledFxGraph, example_inputs: List[torch.Tensor]):
        """
        Store a serialized CompiledFxGraph on disk.
        """
        disk_compiled_graph = copy(compiled_graph)
        disk_compiled_graph.compiled_artifact = None
        shape_env = FxGraphCache._get_shape_env()
        symints = FxGraphCache._filter_symints(example_inputs)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(symints)
        content = pickle.dumps(disk_compiled_graph)
        subdir = FxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, sha256_hash(content))
        write_atomic(path, content)

    @staticmethod
    def load(compile_fx_fn: Callable[..., Any], gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fx_kwargs: Dict[str, Any]):
        """
        Load a compiled graph from the cache. If a cached entry does not exist,
        compile the graph and save it to the cache.
        """
        from filelock import FileLock
        key = compiled_fx_graph_hash(gm, example_inputs, fx_kwargs)
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
        with lock:
            compiled_graph = FxGraphCache._lookup_graph(key, example_inputs)
            if compiled_graph is None:
                log.debug('fx graph cache miss for key %s', key)
                counters['inductor']['fxgraph_cache_miss'] += 1
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)
                FxGraphCache._save_graph(key, compiled_graph, example_inputs)
            else:
                log.debug('fx graph cache hit for key %s', key)
                counters['inductor']['fxgraph_cache_hit'] += 1
            return compiled_graph

    @staticmethod
    def clear():
        """
        Clear out the on-disk cache.
        """
        shutil.rmtree(FxGraphCache._get_tmp_dir())