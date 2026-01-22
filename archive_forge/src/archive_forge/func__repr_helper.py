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
def _repr_helper(self, formatter):
    arguments = [f'input={formatter(self.input)}', f'args={formatter(self.args)}', f'kwargs={formatter(self.kwargs)}', f'broadcasts_input={self.broadcasts_input}', f'name={repr(self.name)}']
    return f'SampleInput({', '.join((a for a in arguments if a is not None))})'