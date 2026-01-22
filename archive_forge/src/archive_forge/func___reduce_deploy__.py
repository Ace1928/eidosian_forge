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
def __reduce_deploy__(self, importer: Importer):
    dict_without_graph = self.__dict__.copy()
    dict_without_graph['_graphmodule_cls_name'] = self.__class__.__name__
    del dict_without_graph['_graph']
    python_code = self.recompile()
    import_block = _format_import_block(python_code.globals, importer)
    return (reduce_deploy_graph_module, (dict_without_graph, import_block))