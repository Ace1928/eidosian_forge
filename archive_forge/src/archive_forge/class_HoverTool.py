from __future__ import annotations
import logging # isort:skip
import difflib
import typing as tp
from math import nan
from typing import Literal
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.validation import error
from ..core.validation.errors import NO_RANGE_TOOL_RANGES
from ..model import Model
from ..util.strings import nice_join
from .annotations import BoxAnnotation, PolyAnnotation, Span
from .callbacks import Callback
from .dom import Template
from .glyphs import (
from .nodes import Node
from .ranges import Range
from .renderers import DataRenderer, GlyphRenderer
from .ui import UIElement
class HoverTool(InspectTool):
    """ *toolbar icon*: |hover_icon|

    The hover tool is a passive inspector tool. It is generally on at all
    times, but can be configured in the inspector's menu associated with the
    *toolbar icon* shown above.

    By default, the hover tool displays informational tooltips whenever the
    cursor is directly over a glyph. The data to show comes from the glyph's
    data source, and what to display is configurable with the ``tooltips``
    property that maps display names to columns in the data source, or to
    special known variables.

    Here is an example of how to configure and use the hover tool::

        # Add tooltip (name, field) pairs to the tool. See below for a
        # description of possible field values.
        hover.tooltips = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("radius", "@radius"),
            ("fill color", "$color[hex, swatch]:fill_color"),
            ("fill color", "$color[hex]:fill_color"),
            ("fill color", "$color:fill_color"),
            ("fill color", "$swatch:fill_color"),
            ("foo", "@foo"),
            ("bar", "@bar"),
            ("baz", "@baz{safe}"),
            ("total", "@total{$0,0.00}"),
        ]

    You can also supply a ``Callback`` to the ``HoverTool``, to build custom
    interactions on hover. In this case you may want to turn the tooltips
    off by setting ``tooltips=None``.

    .. warning::
        When supplying a callback or custom template, the explicit intent
        of this Bokeh Model is to embed *raw HTML and  JavaScript code* for
        a browser to execute. If any part of the code is derived from untrusted
        user inputs, then you must take appropriate care to sanitize the user
        input prior to passing to Bokeh.

    Hover tool does not currently work with the following glyphs:

    .. hlist::
        :columns: 3

        * annulus
        * arc
        * bezier
        * image_url
        * oval
        * patch
        * quadratic
        * ray
        * step
        * text

    .. |hover_icon| image:: /_images/icons/Hover.png
        :height: 24px
        :alt: Icon of a popup tooltip with abstract lines of text representing the hover tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = Either(Auto, List(Instance(DataRenderer)), default='auto', help='\n    A list of renderers to hit test against. If unset, defaults to\n    all renderers on a plot.\n    ')
    callback = Nullable(Instance(Callback), help="\n    A callback to run in the browser whenever the input's value changes. The\n    ``cb_data`` parameter that is available to the Callback code will contain two\n    ``HoverTool`` specific fields:\n\n    :index: object containing the indices of the hovered points in the data source\n    :geometry: object containing the coordinates of the hover cursor\n    ")
    tooltips = Either(Null, Instance(Template), String, List(Tuple(String, String)), default=[('index', '$index'), ('data (x, y)', '($x, $y)'), ('screen (x, y)', '($sx, $sy)')], help='\n    The (name, field) pairs describing what the hover tool should\n    display when there is a hit.\n\n    Field names starting with "@" are interpreted as columns on the\n    data source. For instance, "@temp" would look up values to display\n    from the "temp" column of the data source.\n\n    Field names starting with "$" are special, known fields:\n\n    :$index: index of hovered point in the data source\n    :$name: value of the ``name`` property of the hovered glyph renderer\n    :$x: x-coordinate under the cursor in data space\n    :$y: y-coordinate under the cursor in data space\n    :$sx: x-coordinate under the cursor in screen (canvas) space\n    :$sy: y-coordinate under the cursor in screen (canvas) space\n    :$color: color data from data source, with the syntax:\n        ``$color[options]:field_name``. The available options\n        are: ``hex`` (to display the color as a hex value), ``swatch``\n        (color data from data source displayed as a small color box)\n    :$swatch: color data from data source displayed as a small color box\n\n    Field names that begin with ``@`` are associated with columns in a\n    ``ColumnDataSource``. For instance the field name ``"@price"`` will\n    display values from the ``"price"`` column whenever a hover is triggered.\n    If the hover is for the 17th glyph, then the hover tooltip will\n    correspondingly display the 17th price value.\n\n    Note that if a column name contains spaces, the it must be supplied by\n    surrounding it in curly braces, e.g. ``@{adjusted close}`` will display\n    values from a column named ``"adjusted close"``.\n\n    Sometimes (especially with stacked charts) it is desirable to allow the\n    name of the column be specified indirectly. The field name ``@$name`` is\n    distinguished in that it will look up the ``name`` field on the hovered\n    glyph renderer, and use that value as the column name. For instance, if\n    a user hovers with the name ``"US East"``, then ``@$name`` is equivalent to\n    ``@{US East}``.\n\n    By default, values for fields (e.g. ``@foo``) are displayed in a basic\n    numeric format. However it is possible to control the formatting of values\n    more precisely. Fields can be modified by appending a format specified to\n    the end in curly braces. Some examples are below.\n\n    .. code-block:: python\n\n        "@foo{0,0.000}"    # formats 10000.1234 as: 10,000.123\n\n        "@foo{(.00)}"      # formats -10000.1234 as: (10000.123)\n\n        "@foo{($ 0.00 a)}" # formats 1230974 as: $ 1.23 m\n\n    Specifying a format ``{safe}`` after a field name will override automatic\n    escaping of the tooltip data source. Any HTML tags in the data tags will\n    be rendered as HTML in the resulting HoverTool output. See\n    :ref:`custom_hover_tooltip` for a more detailed example.\n\n    ``None`` is also a valid value for tooltips. This turns off the\n    rendering of tooltips. This is mostly useful when supplying other\n    actions on hover via the callback property.\n\n    .. note::\n        The tooltips attribute can also be configured with a mapping type,\n        e.g. ``dict`` or ``OrderedDict``.\n\n    ').accepts(Dict(String, String), lambda d: list(d.items()))
    formatters = Dict(String, Either(Enum(TooltipFieldFormatter), Instance(CustomJSHover)), default=lambda: dict(), help='\n    Specify the formatting scheme for data source columns, e.g.\n\n    .. code-block:: python\n\n        tool.formatters = {"@date": "datetime"}\n\n    will cause format specifications for the "date" column to be interpreted\n    according to the "datetime" formatting scheme. The following schemes are\n    available:\n\n    :"numeral":\n        Provides a wide variety of formats for numbers, currency, bytes, times,\n        and percentages. The full set of formats can be found in the\n        |NumeralTickFormatter| reference documentation.\n\n    :"datetime":\n        Provides formats for date and time values. The full set of formats is\n        listed in the |DatetimeTickFormatter| reference documentation.\n\n    :"printf":\n        Provides formats similar to C-style "printf" type specifiers. See the\n        |PrintfTickFormatter| reference documentation for complete details.\n\n    If no formatter is specified for a column name, the default ``"numeral"``\n    formatter is assumed.\n\n    .. |NumeralTickFormatter| replace:: :class:`~bokeh.models.formatters.NumeralTickFormatter`\n    .. |DatetimeTickFormatter| replace:: :class:`~bokeh.models.formatters.DatetimeTickFormatter`\n    .. |PrintfTickFormatter| replace:: :class:`~bokeh.models.formatters.PrintfTickFormatter`\n\n    ')
    mode = Enum('mouse', 'hline', 'vline', help='\n    Whether to consider hover pointer as a point (x/y values), or a\n    span on h or v directions.\n    ')
    muted_policy = Enum('show', 'ignore', default='show', help='\n    Whether to avoid showing tooltips on muted glyphs.\n    ')
    point_policy = Enum('snap_to_data', 'follow_mouse', 'none', help='\n    Whether the tooltip position should snap to the "center" (or other anchor)\n    position of the associated glyph, or always follow the current mouse cursor\n    position.\n    ')
    line_policy = Enum('prev', 'next', 'nearest', 'interp', 'none', default='nearest', help='\n    Specifies where the tooltip will be positioned when hovering over line\n    glyphs:\n\n    :"prev": between the nearest two adjacent line points, positions the\n        tooltip at the point with the lower ("previous") index\n    :"next": between the nearest two adjacent line points, positions the\n        tooltip at the point with the higher ("next") index\n    :"nearest": between the nearest two adjacent line points, positions the\n        tooltip on the point that is nearest to the mouse cursor location\n    :"interp": positions the tooltip at an interpolated point on the segment\n        joining the two nearest adjacent line points.\n    :"none": positions the tooltip directly under the mouse cursor location\n\n    ')
    anchor = Enum(Anchor, default='center', help='\n    If point policy is set to `"snap_to_data"`, `anchor` defines the attachment\n    point of a tooltip. The default is to attach to the center of a glyph.\n    ')
    attachment = Enum(TooltipAttachment, help='\n    Whether the tooltip should be displayed to the left or right of the cursor\n    position or above or below it, or if it should be automatically placed\n    in the horizontal or vertical dimension.\n    ')
    show_arrow = Bool(default=True, help="\n    Whether tooltip's arrow should be shown.\n    ")