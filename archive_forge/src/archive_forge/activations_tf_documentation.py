import math
import tensorflow as tf
from packaging.version import parse

    Gated Linear Unit. Implementation as defined in the original paper (see https://arxiv.org/abs/1612.08083), where
    the input `x` is split in two halves across a dimension (`axis`), A and B, returning A * sigmoid(B).

    Args:
        `x`: float Tensor to perform activation
        `axis`: dimension across which `x` be split in half

    Returns:
        `x` with the GLU activation applied (with its size halved across the dimension `axis`).
    