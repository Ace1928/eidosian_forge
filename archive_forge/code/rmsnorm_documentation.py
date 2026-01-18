from typing import Optional
import torch
from torch import nn
from .. import _is_triton_available

        An addition fused with forward.

            z = layer.increment_and_forward_(x, y)

        is equivalent to

            x += y
            z = layer(x)
        