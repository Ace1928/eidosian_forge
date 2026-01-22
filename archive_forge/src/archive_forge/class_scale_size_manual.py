from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@document
class scale_size_manual(_scale_manual):
    """
    Custom discrete size scale

    Parameters
    ----------
    values : array_like | dict
        Sizes that make up the palette. The values will be matched
        with the `limits` of the scale or the `breaks` if provided.
        If it is a dict then it should map data values to sizes.
    {superclass_parameters}
    """
    _aesthetics = ['size']