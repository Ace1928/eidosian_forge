import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def unconvert_from_RGB_255(colors):
    """
    Return a tuple where each element gets divided by 255

    Takes a (list of) color tuple(s) where each element is between 0 and
    255. Returns the same tuples where each tuple element is normalized to
    a value between 0 and 1
    """
    return (colors[0] / 255.0, colors[1] / 255.0, colors[2] / 255.0)