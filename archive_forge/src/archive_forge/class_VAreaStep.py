from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class VAreaStep(FillGlyph, HatchGlyph):
    """ Render a vertically directed area between two equal length sequences
    of y-coordinates with the same x-coordinates using step lines.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VAreaStep.py'
    _args = ('x', 'y1', 'y2')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the points of the area.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates for the points of one side of the area.\n    ')
    y2 = NumberSpec(default=field('y2'), help='\n    The y-coordinates for the points of the other side of the area.\n    ')
    step_mode = Enum(StepMode, default='before', help='\n    Where the step "level" should be drawn in relation to the x and y\n    coordinates. The parameter can assume one of three values:\n\n    * ``before``: (default) Draw step levels before each x-coordinate (no step before the first point)\n    * ``after``:  Draw step levels after each x-coordinate (no step after the last point)\n    * ``center``: Draw step levels centered on each x-coordinate\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the vertical directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the vertical directed area.\n    ')