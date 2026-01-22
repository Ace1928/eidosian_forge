import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
@staticmethod
def _linear_node_to_module_and_attribute_name(graph_module, linear_node_target):
    names = linear_node_target.split('.')
    mod = graph_module
    if len(names) > 1:
        for name in names[:-1]:
            mod = getattr(mod, name)
    return (mod, names[-1])