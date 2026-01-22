from __future__ import annotations
import logging # isort:skip
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
class DataRange1d(DataRange):
    """ An auto-fitting range in a continuous scalar dimension.

    By default the ``start`` and ``end`` of the range automatically
    assume min and max values of the data for associated renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get('follow') is not None:
            kwargs['bounds'] = None
        super().__init__(*args, **kwargs)
    range_padding = Either(Float, TimeDelta, default=0.1, help='\n    How much padding to add around the computed data bounds.\n\n    When ``range_padding_units`` is set to ``"percent"``, the span of the\n    range span is expanded to make the range ``range_padding`` percent larger.\n\n    When ``range_padding_units`` is set to ``"absolute"``, the start and end\n    of the range span are extended by the amount ``range_padding``.\n    ')
    range_padding_units = Enum(PaddingUnits, default='percent', help='\n    Whether the ``range_padding`` should be interpreted as a percentage, or\n    as an absolute quantity. (default: ``"percent"``)\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=True), help="\n    The bounds that the range is allowed to go to. Typically used to prevent\n    the user from panning/zooming/etc away from the data.\n\n    By default, the bounds will be None, allowing your plot to pan/zoom as far\n    as you want. If bounds are 'auto' they will be computed to be the same as\n    the start and end of the ``DataRange1d``.\n\n    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether\n    your range is increasing or decreasing, the first item should be the\n    minimum value of the range and the second item should be the maximum.\n    Setting ``min > max`` will result in a ``ValueError``.\n\n    If you only want to constrain one end of the plot, you can set ``min`` or\n    ``max`` to ``None`` e.g. ``DataRange1d(bounds=(None, 12))``\n    ")
    min_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval. If set to ``None`` (default), the minimum\n    interval is not bound.')
    max_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval. Note that ``bounds`` can impose an\n    implicit constraint on the maximum interval as well.')
    flipped = Bool(default=False, help='\n    Whether the range should be "flipped" from its normal direction when\n    auto-ranging.\n    ')
    follow = Nullable(Enum(StartEnd), help='\n    Configure the data to follow one or the other data extreme, with a\n    maximum range size of ``follow_interval``.\n\n    If set to ``"start"`` then the range will adjust so that ``start`` always\n    corresponds to the minimum data value (or maximum, if ``flipped`` is\n    ``True``).\n\n    If set to ``"end"`` then the range will adjust so that ``end`` always\n    corresponds to the maximum data value (or minimum, if ``flipped`` is\n    ``True``).\n\n    If set to ``None`` (default), then auto-ranging does not follow, and\n    the range will encompass both the minimum and maximum data values.\n\n    ``follow`` cannot be used with bounds, and if set, bounds will be set to\n    ``None``.\n    ')
    follow_interval = Nullable(Either(Float, TimeDelta), help='\n    If ``follow`` is set to ``"start"`` or ``"end"`` then the range will\n    always be constrained to that::\n\n         abs(r.start - r.end) <= follow_interval\n\n    is maintained.\n\n    ')
    default_span = Either(Float, TimeDelta, default=2.0, help='\n    A default width for the interval, in case ``start`` is equal to ``end``\n    (if used with a log axis, default_span is in powers of 10).\n    ')
    only_visible = Bool(default=False, help='\n    If True, renderers that that are not visible will be excluded from automatic\n    bounds computations.\n    ')