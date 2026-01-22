from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
@abstract
class Interpolator(Transform):
    """ Base class for interpolator transforms.

    Interpolators return the value of a function which has been evaluated
    between specified (x, y) pairs of data.  As an example, if two control
    point pairs were provided to the interpolator, a linear interpolaction
    at a specific value of 'x' would result in the value of 'y' which existed
    on the line connecting the two control points.

    The control point pairs for the interpolators can be specified through either

    * A literal sequence of values:

    .. code-block:: python

        interp = Interpolator(x=[1, 2, 3, 4, 5], y=[2, 5, 10, 12, 16])

    * or a pair of columns defined in a ``ColumnDataSource`` object:

    .. code-block:: python

        interp = Interpolator(x="year", y="earnings", data=jewlery_prices))


    This is the base class and is not intended to end use.  Please see the
    documentation for the final derived classes (``Jitter``, ``LineraInterpolator``,
    ``StepInterpolator``) for more information on their specific methods of
    interpolation.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x = Required(Either(String, Seq(Float)), help='\n    Independent coordinate denoting the location of a point.\n    ')
    y = Required(Either(String, Seq(Float)), help='\n    Dependant coordinate denoting the value of a point at a location.\n    ')
    data = Nullable(Instance(ColumnarDataSource), help='\n    Data which defines the source for the named columns if a string is passed to either the ``x`` or ``y`` parameters.\n    ')
    clip = Bool(True, help='\n    Determine if the interpolation should clip the result to include only values inside its predefined range.\n    If this is set to False, it will return the most value of the closest point.\n    ')