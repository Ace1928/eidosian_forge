from __future__ import annotations
from typing import ClassVar, List
import param
from bokeh.models import Div as BkDiv, Spacer as BkSpacer
from ..io.resources import CDN_DIST
from ..reactive import Reactive
class VSpacer(Spacer):
    """
    The `VSpacer` layout provides responsive vertical spacing.

    Using this component we can space objects equidistantly in a layout and
    allow the empty space to shrink when the browser is resized.

    Reference: https://panel.holoviz.org/user_guide/Customization.html#spacers

    :Example:

    >>> pn.Column(
    ...     pn.layout.VSpacer(), 'Item 1',
    ...     pn.layout.VSpacer(), 'Item 2',
    ...     pn.layout.VSpacer()
    ... )
    """
    sizing_mode = param.Parameter(default='stretch_height', readonly=True)