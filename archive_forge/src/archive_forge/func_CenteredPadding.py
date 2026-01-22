from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
@staticmethod
def CenteredPadding(interval, size, left_justify=True):
    """Compute information for centering a string in a fixed space.

    Given two integers interval and size, with size <= interval, this
    function computes two integers left_padding and right_padding with
      left_padding + right_padding + size = interval
    and
      |left_padding - right_padding| <= 1.

    In the case that interval and size have different parity,
    left_padding will be larger iff left_justify is True. (That is,
    iff the string should be left justified in the "center" space.)

    Args:
      interval: Size of the fixed space.
      size: Size of the string to center in that space.
      left_justify: (optional, default: True) Whether the string
        should be left-justified in the center space.

    Returns:
      left_padding, right_padding: The size of the left and right
        margins for centering the string.

    Raises:
      FormatterException: If size > interval.
    """
    if size > interval:
        raise FormatterException('Illegal state in table formatting')
    same_parity = interval % 2 == size % 2
    padding = (interval - size) // 2
    if same_parity:
        return (padding, padding)
    elif left_justify:
        return (padding, padding + 1)
    else:
        return (padding + 1, padding)