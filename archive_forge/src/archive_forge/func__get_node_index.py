import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def _get_node_index(tree: Dict[str, Any], tree_index: Optional[int]) -> str:
    tree_num = f'{tree_index}-' if tree_index is not None else ''
    is_split = _is_split_node(tree)
    node_type = 'S' if is_split else 'L'
    node_num = tree.get('split_index' if is_split else 'leaf_index', 0)
    return f'{tree_num}{node_type}{node_num}'