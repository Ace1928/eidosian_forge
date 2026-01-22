from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@document
class scale_shape_manual(_scale_manual):
    """
    Custom discrete shape scale

    Parameters
    ----------
    values : array_like | dict
        Shapes that make up the palette. See
        [](`matplotlib.markers`) for list of all possible
        shapes. The values will be matched with the `limits`
        of the scale or the `breaks` if provided.
        If it is a dict then it should map data values to shapes.
    {superclass_parameters}

    See Also
    --------
    [](`matplotlib.markers`)
    """
    _aesthetics = ['shape']