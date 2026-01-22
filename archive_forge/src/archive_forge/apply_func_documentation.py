from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning_fabric.utilities.types import _DEVICE
Recursively walk through a collection and convert single-item tensors to scalar values.

    Raises:
        ValueError:
            If tensors inside ``metrics`` contains multiple elements, hence preventing conversion to a scalar.

    