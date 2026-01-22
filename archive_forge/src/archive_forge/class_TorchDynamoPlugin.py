import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class TorchDynamoPlugin(KwargsHandler):
    """
    This plugin is used to compile a model with PyTorch 2.0
    """
    backend: DynamoBackend = field(default=None, metadata={'help': f'Possible options are {[b.value.lower() for b in DynamoBackend]}'})
    mode: str = field(default=None, metadata={'help': "Possible options are 'default', 'reduce-overhead' or 'max-autotune'"})
    fullgraph: bool = field(default=None, metadata={'help': 'Whether it is ok to break model into several subgraphs'})
    dynamic: bool = field(default=None, metadata={'help': 'Whether to use dynamic shape for tracing'})
    options: Any = field(default=None, metadata={'help': 'A dictionary of options to pass to the backend.'})
    disable: bool = field(default=False, metadata={'help': 'Turn torch.compile() into a no-op for testing'})

    def __post_init__(self):
        prefix = 'ACCELERATE_DYNAMO_'
        if self.backend is None:
            self.backend = os.environ.get(prefix + 'BACKEND', 'no')
        self.backend = DynamoBackend(self.backend.upper())
        if self.mode is None:
            self.mode = os.environ.get(prefix + 'MODE', 'default')
        if self.fullgraph is None:
            self.fullgraph = str_to_bool(os.environ.get(prefix + 'USE_FULLGRAPH', 'False')) == 1
        if self.dynamic is None:
            self.dynamic = str_to_bool(os.environ.get(prefix + 'USE_DYNAMIC', 'False')) == 1

    def to_dict(self):
        dynamo_config = copy.deepcopy(self.__dict__)
        dynamo_config['backend'] = dynamo_config['backend'].value.lower()
        return dynamo_config