from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..model import Model
from ..types import Floats2d
from ..util import get_width
def _get_d_moments(ops: Ops, dy: Floats2d, X: Floats2d, mu: Floats2d) -> Tuple[Floats2d, Floats2d, Floats2d]:
    dist = X - mu
    return (dist, ops.xp.sum(dy, axis=1, keepdims=True), ops.xp.sum(dy * dist, axis=1, keepdims=True))