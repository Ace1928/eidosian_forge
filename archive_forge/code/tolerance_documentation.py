from typing import Union, Iterable, TYPE_CHECKING
import numpy as np
Checks if the tensor's elements are all near multiples of the period.

    Args:
        a: Tensor of elements that could all be near multiples of the period.
        period: The period, e.g. 2 pi when working in radians.
        atol: Absolute tolerance.
    