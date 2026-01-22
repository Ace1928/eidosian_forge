import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum
class UnionFind:

    def __init__(self, n):
        self.parent: List[Optional[int]] = [None] * n
        self.size: List[int] = [0] * n

    def make_set(self, v: int):
        self.parent[v] = v
        self.size[v] = 1

    def find(self, v: int) -> int:
        par = self.parent[v]
        if v == par:
            return v
        assert par is not None
        self.parent[v] = self.find(par)
        return cast(int, self.parent[v])

    def join(self, a: int, b: int):
        a, b = (self.find(a), self.find(b))
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            a, b = (b, a)
        self.parent[b] = a
        self.size[a] += self.size[b]