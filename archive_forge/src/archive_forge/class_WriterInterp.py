import argparse
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, Callable, Dict, Union
import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import (
from torch._dynamo.utils import clone_inputs, counters, same
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.hub import tqdm
from .. import config
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
class WriterInterp(fx.Interpreter):

    def __init__(self, mod, subdir):
        super().__init__(mod)
        self.subdir = subdir

    def run_node(self, n):
        r = super().run_node(n)
        name = n.name
        if name in known_names:
            pbar.update(1)
            writer.write_tensor(os.path.join(self.subdir, name), r)
        return r