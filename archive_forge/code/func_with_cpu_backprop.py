from typing import Any, Callable, Tuple
import numpy
from thinc.backends import Ops
from ..config import registry
from ..model import Model
def with_cpu_backprop(d_outputs):
    cpu_d_outputs = _to_cpu(d_outputs)
    return backprop(cpu_d_outputs)