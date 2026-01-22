from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import get_width
from .noop import noop
def _apply_attention(ops, attention, X, lengths):
    output = X * attention

    def apply_attention_bwd(d_output):
        d_attention = (X * d_output).sum(axis=1, keepdims=True)
        dX = d_output * attention
        return (dX, d_attention)
    return (output, apply_attention_bwd)