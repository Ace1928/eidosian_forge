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
@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """
    compiled_artifact: Optional[Callable[..., Any]] = None
    current_callable: Optional[Callable[..., Any]] = None
    cache_key: Optional[str] = None
    artifact_path: Optional[str] = None
    cache_linemap: Optional[List[Tuple[int, str]]] = None
    device_types: Set[str] = field(default_factory=set)
    device_idxs: Set[int] = field(default_factory=set)
    mutated_inputs: Set[str] = field(default_factory=set)
    mutated_input_idxs: Set[int] = field(default_factory=set)
    constants: Dict[str, torch.Tensor] = field(default_factory=dict)
    output_strides: Optional[List[Optional[Tuple[int, ...]]]] = None
    guards_expr: Optional[str] = None
    _boxed_call: Optional[bool] = None

    def __init__(self, compiled_artifact: Optional[Callable[..., Any]], graph: GraphLowering, output_strides: List[Optional[Tuple[int, ...]]]):
        self.compiled_artifact = compiled_artifact
        self.cache_key = graph.cache_key
        self.artifact_path = graph.cache_path
        self.cache_linemap = graph.cache_linemap
        self.device_types = graph.device_types
        self.device_idxs = graph.device_idxs
        self.mutated_inputs = graph.mutated_inputs
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        self.constants = graph.constants
        self.output_strides = output_strides
        self.guards_expr = None

    def __call__(self, inputs: List[Any]) -> Any:
        return self.get_current_callable()(inputs)

    def get_current_callable(self) -> Callable[..., Any]:
        if self.current_callable is None:
            return functools.partial(_run_from_cache, weakref.proxy(self))
        else:
            return self.current_callable