import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def validate_scale_values(scale):
    """
    Validates scale values from a colorscale

    :param (list) scale: a strictly increasing list of floats that begins
        with 0 and ends with 1. Its usage derives from a colorscale which is
        a list of two-lists (a list with two elements) of the form
        [value, color] which are used to determine how interpolation weighting
        works between the colors in the colorscale. Therefore scale is just
        the extraction of these values from the two-lists in order
    """
    if len(scale) < 2:
        raise exceptions.PlotlyError('You must input a list of scale values that has at least two values.')
    if scale[0] != 0 or scale[-1] != 1:
        raise exceptions.PlotlyError('The first and last number in your scale must be 0.0 and 1.0 respectively.')
    if not all((x < y for x, y in zip(scale, scale[1:]))):
        raise exceptions.PlotlyError("'scale' must be a list that contains a strictly increasing sequence of numbers.")