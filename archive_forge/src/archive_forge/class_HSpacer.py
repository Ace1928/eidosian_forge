from __future__ import annotations
from typing import ClassVar, List
import param
from bokeh.models import Div as BkDiv, Spacer as BkSpacer
from ..io.resources import CDN_DIST
from ..reactive import Reactive
class HSpacer(Spacer):
    """
    The `HSpacer` layout provides responsive horizontal spacing.

    Using this component we can space objects equidistantly in a layout and
    allow the empty space to shrink when the browser is resized.

    How-to: https://panel.holoviz.org/how_to/layout/spacing.html#spacer-components

    :Example:

    >>> pn.Row(
    ...     pn.layout.HSpacer(), 'Item 1',
    ...     pn.layout.HSpacer(), 'Item 2',
    ...     pn.layout.HSpacer()
    ... )
    """
    sizing_mode = param.Parameter(default='stretch_width', readonly=True)