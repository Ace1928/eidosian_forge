import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
def _encode_signature(self, angle: Union[str, Angle, dict, AngleDatum, AngleValue, UndefinedType]=Undefined, color: Union[str, Color, dict, ColorDatum, ColorValue, UndefinedType]=Undefined, column: Union[str, Column, dict, UndefinedType]=Undefined, description: Union[str, Description, dict, DescriptionValue, UndefinedType]=Undefined, detail: Union[str, Detail, dict, list, UndefinedType]=Undefined, facet: Union[str, Facet, dict, UndefinedType]=Undefined, fill: Union[str, Fill, dict, FillDatum, FillValue, UndefinedType]=Undefined, fillOpacity: Union[str, FillOpacity, dict, FillOpacityDatum, FillOpacityValue, UndefinedType]=Undefined, href: Union[str, Href, dict, HrefValue, UndefinedType]=Undefined, key: Union[str, Key, dict, UndefinedType]=Undefined, latitude: Union[str, Latitude, dict, LatitudeDatum, UndefinedType]=Undefined, latitude2: Union[str, Latitude2, dict, Latitude2Datum, Latitude2Value, UndefinedType]=Undefined, longitude: Union[str, Longitude, dict, LongitudeDatum, UndefinedType]=Undefined, longitude2: Union[str, Longitude2, dict, Longitude2Datum, Longitude2Value, UndefinedType]=Undefined, opacity: Union[str, Opacity, dict, OpacityDatum, OpacityValue, UndefinedType]=Undefined, order: Union[str, Order, dict, list, OrderValue, UndefinedType]=Undefined, radius: Union[str, Radius, dict, RadiusDatum, RadiusValue, UndefinedType]=Undefined, radius2: Union[str, Radius2, dict, Radius2Datum, Radius2Value, UndefinedType]=Undefined, row: Union[str, Row, dict, UndefinedType]=Undefined, shape: Union[str, Shape, dict, ShapeDatum, ShapeValue, UndefinedType]=Undefined, size: Union[str, Size, dict, SizeDatum, SizeValue, UndefinedType]=Undefined, stroke: Union[str, Stroke, dict, StrokeDatum, StrokeValue, UndefinedType]=Undefined, strokeDash: Union[str, StrokeDash, dict, StrokeDashDatum, StrokeDashValue, UndefinedType]=Undefined, strokeOpacity: Union[str, StrokeOpacity, dict, StrokeOpacityDatum, StrokeOpacityValue, UndefinedType]=Undefined, strokeWidth: Union[str, StrokeWidth, dict, StrokeWidthDatum, StrokeWidthValue, UndefinedType]=Undefined, text: Union[str, Text, dict, TextDatum, TextValue, UndefinedType]=Undefined, theta: Union[str, Theta, dict, ThetaDatum, ThetaValue, UndefinedType]=Undefined, theta2: Union[str, Theta2, dict, Theta2Datum, Theta2Value, UndefinedType]=Undefined, tooltip: Union[str, Tooltip, dict, list, TooltipValue, UndefinedType]=Undefined, url: Union[str, Url, dict, UrlValue, UndefinedType]=Undefined, x: Union[str, X, dict, XDatum, XValue, UndefinedType]=Undefined, x2: Union[str, X2, dict, X2Datum, X2Value, UndefinedType]=Undefined, xError: Union[str, XError, dict, XErrorValue, UndefinedType]=Undefined, xError2: Union[str, XError2, dict, XError2Value, UndefinedType]=Undefined, xOffset: Union[str, XOffset, dict, XOffsetDatum, XOffsetValue, UndefinedType]=Undefined, y: Union[str, Y, dict, YDatum, YValue, UndefinedType]=Undefined, y2: Union[str, Y2, dict, Y2Datum, Y2Value, UndefinedType]=Undefined, yError: Union[str, YError, dict, YErrorValue, UndefinedType]=Undefined, yError2: Union[str, YError2, dict, YError2Value, UndefinedType]=Undefined, yOffset: Union[str, YOffset, dict, YOffsetDatum, YOffsetValue, UndefinedType]=Undefined):
    """Parameters
    ----------

    angle : str, :class:`Angle`, Dict, :class:`AngleDatum`, :class:`AngleValue`
        Rotation angle of point and text marks.
    color : str, :class:`Color`, Dict, :class:`ColorDatum`, :class:`ColorValue`
        Color of the marks – either fill or stroke color based on  the ``filled`` property
        of mark definition. By default, ``color`` represents fill color for ``"area"``,
        ``"bar"``, ``"tick"``, ``"text"``, ``"trail"``, ``"circle"``, and ``"square"`` /
        stroke color for ``"line"`` and ``"point"``.

        **Default value:** If undefined, the default color depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* 1) For fine-grained control over both fill and stroke colors of the marks,
        please use the ``fill`` and ``stroke`` channels. The ``fill`` or ``stroke``
        encodings have higher precedence than ``color``, thus may override the ``color``
        encoding if conflicting encodings are specified. 2) See the scale documentation for
        more information about customizing `color scheme
        <https://vega.github.io/vega-lite/docs/scale.html#scheme>`__.
    column : str, :class:`Column`, Dict
        A field definition for the horizontal facet of trellis plots.
    description : str, :class:`Description`, Dict, :class:`DescriptionValue`
        A text description of this mark for ARIA accessibility (SVG output only). For SVG
        output the ``"aria-label"`` attribute will be set to this description.
    detail : str, :class:`Detail`, Dict, List
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    facet : str, :class:`Facet`, Dict
        A field definition for the (flexible) facet of trellis plots.

        If either ``row`` or ``column`` is specified, this channel will be ignored.
    fill : str, :class:`Fill`, Dict, :class:`FillDatum`, :class:`FillValue`
        Fill color of the marks. **Default value:** If undefined, the default color depends
        on `mark config <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__
        's ``color`` property.

        *Note:* The ``fill`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    fillOpacity : str, :class:`FillOpacity`, Dict, :class:`FillOpacityDatum`, :class:`FillOpacityValue`
        Fill opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``fillOpacity`` property.
    href : str, :class:`Href`, Dict, :class:`HrefValue`
        A URL to load upon mouse click.
    key : str, :class:`Key`, Dict
        A data field to use as a unique key for data binding. When a visualization’s data is
        updated, the key value will be used to match data elements to existing mark
        instances. Use a key channel to enable object constancy for transitions over dynamic
        data.
    latitude : str, :class:`Latitude`, Dict, :class:`LatitudeDatum`
        Latitude position of geographically projected marks.
    latitude2 : str, :class:`Latitude2`, Dict, :class:`Latitude2Datum`, :class:`Latitude2Value`
        Latitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    longitude : str, :class:`Longitude`, Dict, :class:`LongitudeDatum`
        Longitude position of geographically projected marks.
    longitude2 : str, :class:`Longitude2`, Dict, :class:`Longitude2Datum`, :class:`Longitude2Value`
        Longitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    opacity : str, :class:`Opacity`, Dict, :class:`OpacityDatum`, :class:`OpacityValue`
        Opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``opacity``
        property.
    order : str, :class:`Order`, Dict, List, :class:`OrderValue`
        Order of the marks.


        * For stacked marks, this ``order`` channel encodes `stack order
          <https://vega.github.io/vega-lite/docs/stack.html#order>`__.
        * For line and trail marks, this ``order`` channel encodes order of data points in
          the lines. This can be useful for creating `a connected scatterplot
          <https://vega.github.io/vega-lite/examples/connected_scatterplot.html>`__. Setting
          ``order`` to ``{"value": null}`` makes the line marks use the original order in
          the data sources.
        * Otherwise, this ``order`` channel encodes layer order of the marks.

        **Note** : In aggregate plots, ``order`` field should be ``aggregate`` d to avoid
        creating additional aggregation grouping.
    radius : str, :class:`Radius`, Dict, :class:`RadiusDatum`, :class:`RadiusValue`
        The outer radius in pixels of arc marks.
    radius2 : str, :class:`Radius2`, Dict, :class:`Radius2Datum`, :class:`Radius2Value`
        The inner radius in pixels of arc marks.
    row : str, :class:`Row`, Dict
        A field definition for the vertical facet of trellis plots.
    shape : str, :class:`Shape`, Dict, :class:`ShapeDatum`, :class:`ShapeValue`
        Shape of the mark.


        #.
        For ``point`` marks the supported values include:   - plotting shapes: ``"circle"``,
        ``"square"``, ``"cross"``, ``"diamond"``, ``"triangle-up"``, ``"triangle-down"``,
        ``"triangle-right"``, or ``"triangle-left"``.   - the line symbol ``"stroke"``   -
        centered directional shapes ``"arrow"``, ``"wedge"``, or ``"triangle"``   - a custom
        `SVG path string
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths>`__ (For correct
        sizing, custom shape paths should be defined within a square bounding box with
        coordinates ranging from -1 to 1 along both the x and y dimensions.)

        #.
        For ``geoshape`` marks it should be a field definition of the geojson data

        **Default value:** If undefined, the default shape depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#point-config>`__ 's ``shape``
        property. ( ``"circle"`` if unset.)
    size : str, :class:`Size`, Dict, :class:`SizeDatum`, :class:`SizeValue`
        Size of the mark.


        * For ``"point"``, ``"square"`` and ``"circle"``, – the symbol size, or pixel area
          of the mark.
        * For ``"bar"`` and ``"tick"`` – the bar and tick's size.
        * For ``"text"`` – the text's font size.
        * Size is unsupported for ``"line"``, ``"area"``, and ``"rect"``. (Use ``"trail"``
          instead of line with varying size)
    stroke : str, :class:`Stroke`, Dict, :class:`StrokeDatum`, :class:`StrokeValue`
        Stroke color of the marks. **Default value:** If undefined, the default color
        depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* The ``stroke`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    strokeDash : str, :class:`StrokeDash`, Dict, :class:`StrokeDashDatum`, :class:`StrokeDashValue`
        Stroke dash of the marks.

        **Default value:** ``[1,0]`` (No dash).
    strokeOpacity : str, :class:`StrokeOpacity`, Dict, :class:`StrokeOpacityDatum`, :class:`StrokeOpacityValue`
        Stroke opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeOpacity`` property.
    strokeWidth : str, :class:`StrokeWidth`, Dict, :class:`StrokeWidthDatum`, :class:`StrokeWidthValue`
        Stroke width of the marks.

        **Default value:** If undefined, the default stroke width depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeWidth`` property.
    text : str, :class:`Text`, Dict, :class:`TextDatum`, :class:`TextValue`
        Text of the ``text`` mark.
    theta : str, :class:`Theta`, Dict, :class:`ThetaDatum`, :class:`ThetaValue`
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : str, :class:`Theta2`, Dict, :class:`Theta2Datum`, :class:`Theta2Value`
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    tooltip : str, :class:`Tooltip`, Dict, List, :class:`TooltipValue`
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : str, :class:`Url`, Dict, :class:`UrlValue`
        The URL of an image mark.
    x : str, :class:`X`, Dict, :class:`XDatum`, :class:`XValue`
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : str, :class:`X2`, Dict, :class:`X2Datum`, :class:`X2Value`
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    xError : str, :class:`XError`, Dict, :class:`XErrorValue`
        Error value of x coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    xError2 : str, :class:`XError2`, Dict, :class:`XError2Value`
        Secondary error value of x coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    xOffset : str, :class:`XOffset`, Dict, :class:`XOffsetDatum`, :class:`XOffsetValue`
        Offset of x-position of the marks
    y : str, :class:`Y`, Dict, :class:`YDatum`, :class:`YValue`
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : str, :class:`Y2`, Dict, :class:`Y2Datum`, :class:`Y2Value`
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    yError : str, :class:`YError`, Dict, :class:`YErrorValue`
        Error value of y coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    yError2 : str, :class:`YError2`, Dict, :class:`YError2Value`
        Secondary error value of y coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    yOffset : str, :class:`YOffset`, Dict, :class:`YOffsetDatum`, :class:`YOffsetValue`
        Offset of y-position of the marks
    """
    ...