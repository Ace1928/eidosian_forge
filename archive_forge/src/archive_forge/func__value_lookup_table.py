from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _value_lookup_table(values: Sequence[int], size: int) -> list[int]:
    """
    Generate a lookup table for finding the closest item in values.
    Lookup returns (index into values)+1

    values -- list of values in ascending order, all < size
    size -- size of lookup table and maximum value

    >>> _value_lookup_table([0, 7, 9], 10)
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    """
    middle_values = [0] + [(values[i] + values[i + 1] + 1) // 2 for i in range(len(values) - 1)] + [size]
    lookup_table = []
    for i in range(len(middle_values) - 1):
        count = middle_values[i + 1] - middle_values[i]
        lookup_table.extend([i] * count)
    return lookup_table