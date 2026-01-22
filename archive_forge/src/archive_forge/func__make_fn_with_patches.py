import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def _make_fn_with_patches(fn, *patches):

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            for module, attr, val in patches:
                stack.enter_context(patch.object(module, attr, val))
            return fn(*args, **kwargs)
    return _fn