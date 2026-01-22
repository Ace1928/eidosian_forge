from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import get_width
from .noop import noop
def _get_attention(ops, Q, key_transform, X, lengths, is_train):
    K, K_bp = key_transform(X, is_train=is_train)
    attention = ops.gemm(K, ops.reshape2f(Q, -1, 1))
    attention = ops.softmax_sequences(attention, lengths)

    def get_attention_bwd(d_attention):
        d_attention = ops.backprop_softmax_sequences(d_attention, attention, lengths)
        dQ = ops.gemm(K, d_attention, trans1=True)
        dY = ops.xp.outer(d_attention, Q)
        dX = K_bp(dY)
        return (dQ, dX)
    return (attention, get_attention_bwd)