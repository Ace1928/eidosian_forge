from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class ImageStack(ImageBase):
    """ Render images given as 3D stacked arrays by flattening each stack into
    an RGBA image using a ``StackColorMapper``.

    The 3D arrays have shape (ny, nx, nstack) where ``nstack`` is the number of
    stacks. The ``color_mapper`` produces an RGBA value for each of the
    (ny, nx) pixels by combining array values in the ``nstack`` direction.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    _args = ('image', 'x', 'y', 'dw', 'dh', 'dilate')
    image = NumberSpec(default=field('image'), help='\n    The 3D arrays of data for the images.\n    ')
    color_mapper = Instance(StackColorMapper, help='\n    ``ScalarColorMapper`` used to map the scalar data from ``image``\n    into RGBA values for display.\n\n    .. note::\n        The color mapping step happens on the client.\n    ')