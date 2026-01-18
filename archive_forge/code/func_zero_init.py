from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
def zero_init(ops: Ops, shape: Shape) -> FloatsXd:
    return ops.alloc_f(shape)