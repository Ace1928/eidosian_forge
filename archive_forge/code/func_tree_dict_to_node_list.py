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
def tree_dict_to_node_list(tree: Dict[str, Any], node_depth: int=1, tree_index: Optional[int]=None, feature_names: Optional[List[str]]=None, parent_node: Optional[str]=None) -> List[Dict[str, Any]]:
    node = create_node_record(tree=tree, node_depth=node_depth, tree_index=tree_index, feature_names=feature_names, parent_node=parent_node)
    res = [node]
    if _is_split_node(tree):
        children = ['left_child', 'right_child']
        for child in children:
            subtree_list = tree_dict_to_node_list(tree=tree[child], node_depth=node_depth + 1, tree_index=tree_index, feature_names=feature_names, parent_node=node['node_index'])
            res.extend(subtree_list)
    return res