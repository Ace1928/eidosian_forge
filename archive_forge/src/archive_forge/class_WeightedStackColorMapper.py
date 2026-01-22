from __future__ import annotations
import logging # isort:skip
from .. import palettes
from ..core.enums import Palette
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH
from ..core.validation.warnings import PALETTE_LENGTH_FACTORS_MISMATCH
from .transforms import Transform
class WeightedStackColorMapper(StackColorMapper):
    """ Maps 3D data arrays of shape ``(ny, nx, nstack)`` to 2D RGBA images
    of shape ``(ny, nx)`` using a palette of length ``nstack``.

    The mapping occurs in two stages. Firstly the RGB values are calculated
    using a weighted sum of the palette colors in the ``nstack`` direction.
    Then the alpha values are calculated using the ``alpha_mapper`` applied to
    the sum of the array in the ``nstack`` direction.

    The RGB values calculated by the ``alpha_mapper`` are ignored by the color
    mapping but are used in any ``ColorBar`` that is displayed.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    alpha_mapper = Instance(ContinuousColorMapper, help='\n    Color mapper used to calculate the alpha values of the mapped data.\n    ')
    color_baseline = Nullable(Float, help='\n    Baseline value used for the weights when calculating the weighted sum of\n    palette colors. If ``None`` then the minimum of the supplied data is used\n    meaning that values at this minimum have a weight of zero and do not\n    contribute to the weighted sum. As a special case, if all data for a\n    particular output pixel are at the color baseline then the color is an\n    evenly weighted average of the colors corresponding to all such values,\n    to avoid the color being undefined.\n    ')
    stack_labels = Nullable(Seq(String), help='\n    An optional sequence of strings to use as labels for the ``nstack`` stacks.\n    If set, the number of labels should match the number of stacks and hence\n    also the number of palette colors.\n\n    The labels are used in hover tooltips for ``ImageStack`` glyphs that use a\n    ``WeightedStackColorMapper`` as their color mapper.\n    ')

    @error(WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH)
    def _check_label_length(self):
        if self.stack_labels is not None:
            nlabel = len(self.stack_labels)
            npalette = len(self.palette)
            if nlabel > npalette:
                self.stack_labels = self.stack_labels[:npalette]
                return f'{nlabel} != {npalette}, removing unwanted stack_labels'
            elif nlabel < npalette:
                self.stack_labels = list(self.stack_labels) + [''] * (npalette - nlabel)
                return f'{nlabel} != {npalette}, padding with empty strings'