import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
class Lit:

    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s