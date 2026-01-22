import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
class _EvalCacheLoader:

    def __init__(self):
        self.eval_cache = {}
        self.next_id = 0

    def cache(self, src: str, globals: Dict[str, Any], co_fields=None):
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """
        key = self._get_key()
        if co_fields:
            key += f' from {co_fields['co_filename']}:{co_fields['co_firstlineno']} in {co_fields['co_name']}'
        self.eval_cache[key] = src
        globals_copy = globals.copy()
        globals_copy['__file__'] = key
        globals_copy['__name__'] = key
        globals_copy['__loader__'] = self
        linecache.lazycache(key, globals_copy)
        return key

    def get_source(self, module_name) -> Optional[str]:
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]
        return None

    def _get_key(self):
        key = f'<eval_with_key>.{self.next_id}'
        self.next_id += 1
        return key