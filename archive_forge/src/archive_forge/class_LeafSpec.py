import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):

    def __new__(cls) -> 'LeafSpec':
        return optree.treespec_leaf(none_is_leaf=True)