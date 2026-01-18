import torch
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator
from typing import Dict, List, Tuple

    A graph transformation that merges matrix multiplication operations that share the same right-hand
    side operand into one large matrix multiplication.
               ____      _________        _________
      ----    |    |    |         |     M|  A * C  |
    M| A  |  T| B  | * K|    C    | =    |---------|
      ---- ,  |    |    |         |     T|  B * C  |
       K       ----      ---------        ---------
                K            R                R
    