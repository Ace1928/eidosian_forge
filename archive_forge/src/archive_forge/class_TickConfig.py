from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TickConfig(AnyMarkConfig):
    """TickConfig schema wrapper

    Parameters
    ----------

    align : dict, :class:`Align`, :class:`ExprRef`, Literal['left', 'center', 'right']
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : dict, float, :class:`ExprRef`
        The rotation angle of the text, in degrees.
    aria : bool, dict, :class:`ExprRef`
        A boolean flag indicating if `ARIA attributes
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ should be
        included (SVG output only). If ``false``, the "aria-hidden" attribute will be set on
        the output SVG element, removing the mark item from the ARIA accessibility tree.
    ariaRole : str, dict, :class:`ExprRef`
        Sets the type of user interface element of the mark item for `ARIA accessibility
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ (SVG output
        only). If specified, this property determines the "role" attribute. Warning: this
        property is experimental and may be changed in the future.
    ariaRoleDescription : str, dict, :class:`ExprRef`
        A human-readable, author-localized description for the role of the mark item for
        `ARIA accessibility
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ (SVG output
        only). If specified, this property determines the "aria-roledescription" attribute.
        Warning: this property is experimental and may be changed in the future.
    aspect : bool, dict, :class:`ExprRef`
        Whether to keep aspect ratio of image marks.
    bandSize : float
        The width of the ticks.

        **Default value:**  3/4 of step (width step for horizontal ticks and height step for
        vertical ticks).
    baseline : str, dict, :class:`ExprRef`, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom']
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : dict, :class:`Blend`, :class:`ExprRef`, Literal[None, 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity']
        The color blend mode for drawing an item on its current background. Any valid `CSS
        mix-blend-mode <https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode>`__
        value can be used.

        __Default value:__ ``"source-over"``
    color : str, dict, :class:`Color`, :class:`ExprRef`, :class:`Gradient`, :class:`HexColor`, :class:`ColorName`, :class:`LinearGradient`, :class:`RadialGradient`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:**


        * This property cannot be used in a `style config
          <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
        * The ``fill`` and ``stroke`` properties have higher precedence than ``color`` and
          will override ``color``.
    cornerRadius : dict, float, :class:`ExprRef`
        The radius in pixels of rounded rectangles or arcs' corners.

        **Default value:** ``0``
    cornerRadiusBottomLeft : dict, float, :class:`ExprRef`
        The radius in pixels of rounded rectangles' bottom left corner.

        **Default value:** ``0``
    cornerRadiusBottomRight : dict, float, :class:`ExprRef`
        The radius in pixels of rounded rectangles' bottom right corner.

        **Default value:** ``0``
    cornerRadiusTopLeft : dict, float, :class:`ExprRef`
        The radius in pixels of rounded rectangles' top right corner.

        **Default value:** ``0``
    cornerRadiusTopRight : dict, float, :class:`ExprRef`
        The radius in pixels of rounded rectangles' top left corner.

        **Default value:** ``0``
    cursor : dict, :class:`Cursor`, :class:`ExprRef`, Literal['auto', 'default', 'none', 'context-menu', 'help', 'pointer', 'progress', 'wait', 'cell', 'crosshair', 'text', 'vertical-text', 'alias', 'copy', 'move', 'no-drop', 'not-allowed', 'e-resize', 'n-resize', 'ne-resize', 'nw-resize', 's-resize', 'se-resize', 'sw-resize', 'w-resize', 'ew-resize', 'ns-resize', 'nesw-resize', 'nwse-resize', 'col-resize', 'row-resize', 'all-scroll', 'zoom-in', 'zoom-out', 'grab', 'grabbing']
        The mouse cursor used over the mark. Any valid `CSS cursor type
        <https://developer.mozilla.org/en-US/docs/Web/CSS/cursor#Values>`__ can be used.
    description : str, dict, :class:`ExprRef`
        A text description of the mark item for `ARIA accessibility
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA>`__ (SVG output
        only). If specified, this property determines the `"aria-label" attribute
        <https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Techniques/Using_the_aria-label_attribute>`__.
    dir : dict, :class:`ExprRef`, Literal['ltr', 'rtl'], :class:`TextDirection`
        The direction of the text. One of ``"ltr"`` (left-to-right) or ``"rtl"``
        (right-to-left). This property determines on which side is truncated in response to
        the limit parameter.

        **Default value:** ``"ltr"``
    dx : dict, float, :class:`ExprRef`
        The horizontal offset, in pixels, between the text label and its anchor point. The
        offset is applied after rotation by the *angle* property.
    dy : dict, float, :class:`ExprRef`
        The vertical offset, in pixels, between the text label and its anchor point. The
        offset is applied after rotation by the *angle* property.
    ellipsis : str, dict, :class:`ExprRef`
        The ellipsis string for text truncated in response to the limit parameter.

        **Default value:** ``"…"``
    endAngle : dict, float, :class:`ExprRef`
        The end angle in radians for arc marks. A value of ``0`` indicates up (north),
        increasing values proceed clockwise.
    fill : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`Gradient`, :class:`HexColor`, :class:`ColorName`, :class:`LinearGradient`, :class:`RadialGradient`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : dict, float, :class:`ExprRef`
        The fill opacity (value between [0,1]).

        **Default value:** ``1``
    filled : bool
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : str, dict, :class:`ExprRef`
        The typeface to set the text in (e.g., ``"Helvetica Neue"`` ).
    fontSize : dict, float, :class:`ExprRef`
        The font size, in pixels.

        **Default value:** ``11``
    fontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        The font style (e.g., ``"italic"`` ).
    fontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        The font weight. This can be either a string (e.g ``"bold"``, ``"normal"`` ) or a
        number ( ``100``, ``200``, ``300``, ..., ``900`` where ``"normal"`` = ``400`` and
        ``"bold"`` = ``700`` ).
    height : dict, float, :class:`ExprRef`
        Height of the marks.
    href : str, dict, :class:`URI`, :class:`ExprRef`
        A URL to load upon mouse click. If defined, the mark acts as a hyperlink.
    innerRadius : dict, float, :class:`ExprRef`
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.

        **Default value:** ``0``
    interpolate : dict, :class:`ExprRef`, :class:`Interpolate`, Literal['basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open', 'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step', 'step-before', 'step-after']
        The line interpolation method to use for line and area marks. One of the following:


        * ``"linear"`` : piecewise linear segments, as in a polyline.
        * ``"linear-closed"`` : close the linear segments to form a polygon.
        * ``"step"`` : alternate between horizontal and vertical segments, as in a step
          function.
        * ``"step-before"`` : alternate between vertical and horizontal segments, as in a
          step function.
        * ``"step-after"`` : alternate between horizontal and vertical segments, as in a
          step function.
        * ``"basis"`` : a B-spline, with control point duplication on the ends.
        * ``"basis-open"`` : an open B-spline; may not intersect the start or end.
        * ``"basis-closed"`` : a closed B-spline, as in a loop.
        * ``"cardinal"`` : a Cardinal spline, with control point duplication on the ends.
        * ``"cardinal-open"`` : an open Cardinal spline; may not intersect the start or end,
          but will intersect other control points.
        * ``"cardinal-closed"`` : a closed Cardinal spline, as in a loop.
        * ``"bundle"`` : equivalent to basis, except the tension parameter is used to
          straighten the spline.
        * ``"monotone"`` : cubic interpolation that preserves monotonicity in y.
    invalid : Literal['filter', None]
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ).


        * If set to ``"filter"`` (default), all data items with null values will be skipped
          (for line, trail, and area marks) or filtered (for other marks).
        * If ``null``, all data items are included. In this case, invalid values will be
          interpreted as zeroes.
    limit : dict, float, :class:`ExprRef`
        The maximum length of the text mark in pixels. The text value will be automatically
        truncated if the rendered size exceeds the limit.

        **Default value:** ``0`` -- indicating no limit
    lineBreak : str, dict, :class:`ExprRef`
        A delimiter, such as a newline character, upon which to break text strings into
        multiple lines. This property is ignored if the text is array-valued.
    lineHeight : dict, float, :class:`ExprRef`
        The line height in pixels (the spacing between subsequent lines of text) for
        multi-line text marks.
    opacity : dict, float, :class:`ExprRef`
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : bool, None
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`, Literal['horizontal', 'vertical']
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical.


        * For bar, rule and tick, this determines whether the size of the bar and tick
          should be applied to x or y dimension.
        * For area, this property determines the orient property of the Vega output.
        * For line and trail marks, this property determines the sort order of the points in
          the line if ``config.sortLineBy`` is not specified. For stacked charts, this is
          always determined by the orientation of the stack; therefore explicitly specified
          value will be ignored.
    outerRadius : dict, float, :class:`ExprRef`
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.

        **Default value:** ``0``
    padAngle : dict, float, :class:`ExprRef`
        The angular padding applied to sides of the arc, in radians.
    radius : dict, float, :class:`ExprRef`
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.

        **Default value:** ``min(plot_width, plot_height)/2``
    radius2 : dict, float, :class:`ExprRef`
        The secondary (inner) radius in pixels of arc marks.

        **Default value:** ``0``
    shape : str, dict, :class:`ExprRef`, :class:`SymbolShape`
        Shape of the point marks. Supported values include:


        * plotting shapes: ``"circle"``, ``"square"``, ``"cross"``, ``"diamond"``,
          ``"triangle-up"``, ``"triangle-down"``, ``"triangle-right"``, or
          ``"triangle-left"``.
        * the line symbol ``"stroke"``
        * centered directional shapes ``"arrow"``, ``"wedge"``, or ``"triangle"``
        * a custom `SVG path string
          <https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths>`__ (For correct
          sizing, custom shape paths should be defined within a square bounding box with
          coordinates ranging from -1 to 1 along both the x and y dimensions.)

        **Default value:** ``"circle"``
    size : dict, float, :class:`ExprRef`
        Default size for marks.


        * For ``point`` / ``circle`` / ``square``, this represents the pixel area of the
          marks. Note that this value sets the area of the symbol; the side lengths will
          increase with the square root of this value.
        * For ``bar``, this represents the band size of the bar, in pixels.
        * For ``text``, this represents the font size, in pixels.

        **Default value:**


        * ``30`` for point, circle, square marks; width/height's ``step``
        * ``2`` for bar marks with discrete dimensions;
        * ``5`` for bar marks with continuous dimensions;
        * ``11`` for text marks.
    smooth : bool, dict, :class:`ExprRef`
        A boolean flag (default true) indicating if the image should be smoothed when
        resized. If false, individual pixels should be scaled directly rather than
        interpolated with smoothing. For SVG rendering, this option may not work in some
        browsers due to lack of standardization.
    startAngle : dict, float, :class:`ExprRef`
        The start angle in radians for arc marks. A value of ``0`` indicates up (north),
        increasing values proceed clockwise.
    stroke : str, dict, None, :class:`Color`, :class:`ExprRef`, :class:`Gradient`, :class:`HexColor`, :class:`ColorName`, :class:`LinearGradient`, :class:`RadialGradient`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : dict, :class:`ExprRef`, :class:`StrokeCap`, Literal['butt', 'round', 'square']
        The stroke cap for line ending style. One of ``"butt"``, ``"round"``, or
        ``"square"``.

        **Default value:** ``"butt"``
    strokeDash : dict, Sequence[float], :class:`ExprRef`
        An array of alternating stroke, space lengths for creating dashed or dotted lines.
    strokeDashOffset : dict, float, :class:`ExprRef`
        The offset (in pixels) into which to begin drawing with the stroke dash array.
    strokeJoin : dict, :class:`ExprRef`, :class:`StrokeJoin`, Literal['miter', 'round', 'bevel']
        The stroke line join method. One of ``"miter"``, ``"round"`` or ``"bevel"``.

        **Default value:** ``"miter"``
    strokeMiterLimit : dict, float, :class:`ExprRef`
        The miter limit at which to bevel a line join.
    strokeOffset : dict, float, :class:`ExprRef`
        The offset in pixels at which to draw the group stroke and fill. If unspecified, the
        default behavior is to dynamically offset stroked groups such that 1 pixel stroke
        widths align with the pixel grid.
    strokeOpacity : dict, float, :class:`ExprRef`
        The stroke opacity (value between [0,1]).

        **Default value:** ``1``
    strokeWidth : dict, float, :class:`ExprRef`
        The stroke width, in pixels.
    tension : dict, float, :class:`ExprRef`
        Depending on the interpolation type, sets the tension parameter (for line and area
        marks).
    text : str, dict, :class:`Text`, Sequence[str], :class:`ExprRef`
        Placeholder text if the ``text`` channel is not specified
    theta : dict, float, :class:`ExprRef`
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : dict, float, :class:`ExprRef`
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    thickness : float
        Thickness of the tick mark.

        **Default value:**  ``1``
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    timeUnitBandSize : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    tooltip : str, bool, dict, None, float, :class:`ExprRef`, :class:`TooltipContent`
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used.
        * If ``tooltip`` is ``{"content": "data"}``, then all fields that appear in the
          highlighted data point will be used.
        * If set to ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : str, dict, :class:`URI`, :class:`ExprRef`
        The URL of the image file for image marks.
    width : dict, float, :class:`ExprRef`
        Width of the marks.
    x : str, dict, float, :class:`ExprRef`
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : str, dict, float, :class:`ExprRef`
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : str, dict, float, :class:`ExprRef`
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : str, dict, float, :class:`ExprRef`
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/TickConfig'}

    def __init__(self, align: Union[dict, '_Parameter', 'SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, angle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, aria: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, ariaRole: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, ariaRoleDescription: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, aspect: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, bandSize: Union[float, UndefinedType]=Undefined, baseline: Union[str, dict, '_Parameter', 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, blend: Union[dict, '_Parameter', 'SchemaBase', Literal[None, 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity'], UndefinedType]=Undefined, color: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, cornerRadius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, cornerRadiusBottomLeft: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, cornerRadiusBottomRight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, cornerRadiusTopLeft: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, cornerRadiusTopRight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, cursor: Union[dict, '_Parameter', 'SchemaBase', Literal['auto', 'default', 'none', 'context-menu', 'help', 'pointer', 'progress', 'wait', 'cell', 'crosshair', 'text', 'vertical-text', 'alias', 'copy', 'move', 'no-drop', 'not-allowed', 'e-resize', 'n-resize', 'ne-resize', 'nw-resize', 's-resize', 'se-resize', 'sw-resize', 'w-resize', 'ew-resize', 'ns-resize', 'nesw-resize', 'nwse-resize', 'col-resize', 'row-resize', 'all-scroll', 'zoom-in', 'zoom-out', 'grab', 'grabbing'], UndefinedType]=Undefined, description: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, dir: Union[dict, '_Parameter', 'SchemaBase', Literal['ltr', 'rtl'], UndefinedType]=Undefined, dx: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, dy: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, ellipsis: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, endAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fill: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, fillOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, filled: Union[bool, UndefinedType]=Undefined, font: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, fontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, height: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, href: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, innerRadius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, interpolate: Union[dict, '_Parameter', 'SchemaBase', Literal['basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open', 'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step', 'step-before', 'step-after'], UndefinedType]=Undefined, invalid: Union[Literal['filter', None], UndefinedType]=Undefined, limit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, lineBreak: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, lineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, opacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, order: Union[bool, None, UndefinedType]=Undefined, orient: Union['SchemaBase', Literal['horizontal', 'vertical'], UndefinedType]=Undefined, outerRadius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, padAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, radius: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, radius2: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, shape: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, size: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, smooth: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, startAngle: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, stroke: Union[str, dict, None, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, strokeCap: Union[dict, '_Parameter', 'SchemaBase', Literal['butt', 'round', 'square'], UndefinedType]=Undefined, strokeDash: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, strokeDashOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, strokeJoin: Union[dict, '_Parameter', 'SchemaBase', Literal['miter', 'round', 'bevel'], UndefinedType]=Undefined, strokeMiterLimit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, strokeOffset: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, strokeOpacity: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, strokeWidth: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, tension: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, text: Union[str, dict, '_Parameter', 'SchemaBase', Sequence[str], UndefinedType]=Undefined, theta: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, theta2: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, thickness: Union[float, UndefinedType]=Undefined, timeUnitBandPosition: Union[float, UndefinedType]=Undefined, timeUnitBandSize: Union[float, UndefinedType]=Undefined, tooltip: Union[str, bool, dict, None, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, url: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, width: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, x: Union[str, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, x2: Union[str, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, y: Union[str, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, y2: Union[str, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(TickConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole, ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize, baseline=baseline, blend=blend, color=color, cornerRadius=cornerRadius, cornerRadiusBottomLeft=cornerRadiusBottomLeft, cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor, description=description, dir=dir, dx=dx, dy=dy, ellipsis=ellipsis, endAngle=endAngle, fill=fill, fillOpacity=fillOpacity, filled=filled, font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight, height=height, href=href, innerRadius=innerRadius, interpolate=interpolate, invalid=invalid, limit=limit, lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity, order=order, orient=orient, outerRadius=outerRadius, padAngle=padAngle, radius=radius, radius2=radius2, shape=shape, size=size, smooth=smooth, startAngle=startAngle, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash, strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin, strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset, strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, tension=tension, text=text, theta=theta, theta2=theta2, thickness=thickness, timeUnitBandPosition=timeUnitBandPosition, timeUnitBandSize=timeUnitBandSize, tooltip=tooltip, url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)