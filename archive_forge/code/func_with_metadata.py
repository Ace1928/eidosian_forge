import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def with_metadata(self, *, output_process_fn_grad=None, broadcasts_input=None, name=None):
    if output_process_fn_grad is not None:
        self.output_process_fn_grad = output_process_fn_grad
    if broadcasts_input is not None:
        self.broadcasts_input = broadcasts_input
    if name is not None:
        self.name = name
    return self