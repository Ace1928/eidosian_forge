from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FacetedEncoding(VegaLiteSchema):
    """FacetedEncoding schema wrapper

    Parameters
    ----------

    angle : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Rotation angle of point and text marks.
    color : dict, :class:`ColorDef`, :class:`FieldOrDatumDefWithConditionDatumDefGradientstringnull`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull`
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
    column : dict, :class:`RowColumnEncodingFieldDef`
        A field definition for the horizontal facet of trellis plots.
    description : dict, :class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`
        A text description of this mark for ARIA accessibility (SVG output only). For SVG
        output the ``"aria-label"`` attribute will be set to this description.
    detail : dict, :class:`FieldDefWithoutScale`, Sequence[dict, :class:`FieldDefWithoutScale`]
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    facet : dict, :class:`FacetEncodingFieldDef`
        A field definition for the (flexible) facet of trellis plots.

        If either ``row`` or ``column`` is specified, this channel will be ignored.
    fill : dict, :class:`ColorDef`, :class:`FieldOrDatumDefWithConditionDatumDefGradientstringnull`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull`
        Fill color of the marks. **Default value:** If undefined, the default color depends
        on `mark config <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__
        's ``color`` property.

        *Note:* The ``fill`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    fillOpacity : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Fill opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``fillOpacity`` property.
    href : dict, :class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`
        A URL to load upon mouse click.
    key : dict, :class:`FieldDefWithoutScale`
        A data field to use as a unique key for data binding. When a visualization’s data is
        updated, the key value will be used to match data elements to existing mark
        instances. Use a key channel to enable object constancy for transitions over dynamic
        data.
    latitude : dict, :class:`DatumDef`, :class:`LatLongDef`, :class:`LatLongFieldDef`
        Latitude position of geographically projected marks.
    latitude2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        Latitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    longitude : dict, :class:`DatumDef`, :class:`LatLongDef`, :class:`LatLongFieldDef`
        Longitude position of geographically projected marks.
    longitude2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        Longitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    opacity : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``opacity``
        property.
    order : dict, :class:`OrderOnlyDef`, :class:`OrderFieldDef`, :class:`OrderValueDef`, Sequence[dict, :class:`OrderFieldDef`]
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
    radius : dict, :class:`PolarDef`, :class:`PositionValueDef`, :class:`PositionDatumDefBase`, :class:`PositionFieldDefBase`
        The outer radius in pixels of arc marks.
    radius2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        The inner radius in pixels of arc marks.
    row : dict, :class:`RowColumnEncodingFieldDef`
        A field definition for the vertical facet of trellis plots.
    shape : dict, :class:`ShapeDef`, :class:`FieldOrDatumDefWithConditionDatumDefstringnull`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull`
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
    size : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Size of the mark.


        * For ``"point"``, ``"square"`` and ``"circle"``, – the symbol size, or pixel area
          of the mark.
        * For ``"bar"`` and ``"tick"`` – the bar and tick's size.
        * For ``"text"`` – the text's font size.
        * Size is unsupported for ``"line"``, ``"area"``, and ``"rect"``. (Use ``"trail"``
          instead of line with varying size)
    stroke : dict, :class:`ColorDef`, :class:`FieldOrDatumDefWithConditionDatumDefGradientstringnull`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull`
        Stroke color of the marks. **Default value:** If undefined, the default color
        depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* The ``stroke`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    strokeDash : dict, :class:`NumericArrayMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumberArray`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray`
        Stroke dash of the marks.

        **Default value:** ``[1,0]`` (No dash).
    strokeOpacity : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Stroke opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeOpacity`` property.
    strokeWidth : dict, :class:`NumericMarkPropDef`, :class:`FieldOrDatumDefWithConditionDatumDefnumber`, :class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`, :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`
        Stroke width of the marks.

        **Default value:** If undefined, the default stroke width depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeWidth`` property.
    text : dict, :class:`TextDef`, :class:`ValueDefWithConditionStringFieldDefText`, :class:`FieldOrDatumDefWithConditionStringDatumDefText`, :class:`FieldOrDatumDefWithConditionStringFieldDefText`
        Text of the ``text`` mark.
    theta : dict, :class:`PolarDef`, :class:`PositionValueDef`, :class:`PositionDatumDefBase`, :class:`PositionFieldDefBase`
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    tooltip : dict, None, :class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`, Sequence[dict, :class:`StringFieldDef`]
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : dict, :class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`
        The URL of an image mark.
    x : dict, :class:`PositionDef`, :class:`PositionDatumDef`, :class:`PositionFieldDef`, :class:`PositionValueDef`
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    xError : dict, :class:`ValueDefnumber`, :class:`SecondaryFieldDef`
        Error value of x coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    xError2 : dict, :class:`ValueDefnumber`, :class:`SecondaryFieldDef`
        Secondary error value of x coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    xOffset : dict, :class:`OffsetDef`, :class:`ScaleDatumDef`, :class:`ScaleFieldDef`, :class:`ValueDefnumber`
        Offset of x-position of the marks
    y : dict, :class:`PositionDef`, :class:`PositionDatumDef`, :class:`PositionFieldDef`, :class:`PositionValueDef`
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : dict, :class:`DatumDef`, :class:`Position2Def`, :class:`PositionValueDef`, :class:`SecondaryFieldDef`
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    yError : dict, :class:`ValueDefnumber`, :class:`SecondaryFieldDef`
        Error value of y coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    yError2 : dict, :class:`ValueDefnumber`, :class:`SecondaryFieldDef`
        Secondary error value of y coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    yOffset : dict, :class:`OffsetDef`, :class:`ScaleDatumDef`, :class:`ScaleFieldDef`, :class:`ValueDefnumber`
        Offset of y-position of the marks
    """
    _schema = {'$ref': '#/definitions/FacetedEncoding'}

    def __init__(self, angle: Union[dict, 'SchemaBase', UndefinedType]=Undefined, color: Union[dict, 'SchemaBase', UndefinedType]=Undefined, column: Union[dict, 'SchemaBase', UndefinedType]=Undefined, description: Union[dict, 'SchemaBase', UndefinedType]=Undefined, detail: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, facet: Union[dict, 'SchemaBase', UndefinedType]=Undefined, fill: Union[dict, 'SchemaBase', UndefinedType]=Undefined, fillOpacity: Union[dict, 'SchemaBase', UndefinedType]=Undefined, href: Union[dict, 'SchemaBase', UndefinedType]=Undefined, key: Union[dict, 'SchemaBase', UndefinedType]=Undefined, latitude: Union[dict, 'SchemaBase', UndefinedType]=Undefined, latitude2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, longitude: Union[dict, 'SchemaBase', UndefinedType]=Undefined, longitude2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, opacity: Union[dict, 'SchemaBase', UndefinedType]=Undefined, order: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, radius: Union[dict, 'SchemaBase', UndefinedType]=Undefined, radius2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, row: Union[dict, 'SchemaBase', UndefinedType]=Undefined, shape: Union[dict, 'SchemaBase', UndefinedType]=Undefined, size: Union[dict, 'SchemaBase', UndefinedType]=Undefined, stroke: Union[dict, 'SchemaBase', UndefinedType]=Undefined, strokeDash: Union[dict, 'SchemaBase', UndefinedType]=Undefined, strokeOpacity: Union[dict, 'SchemaBase', UndefinedType]=Undefined, strokeWidth: Union[dict, 'SchemaBase', UndefinedType]=Undefined, text: Union[dict, 'SchemaBase', UndefinedType]=Undefined, theta: Union[dict, 'SchemaBase', UndefinedType]=Undefined, theta2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, tooltip: Union[dict, None, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, url: Union[dict, 'SchemaBase', UndefinedType]=Undefined, x: Union[dict, 'SchemaBase', UndefinedType]=Undefined, x2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, xError: Union[dict, 'SchemaBase', UndefinedType]=Undefined, xError2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, xOffset: Union[dict, 'SchemaBase', UndefinedType]=Undefined, y: Union[dict, 'SchemaBase', UndefinedType]=Undefined, y2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, yError: Union[dict, 'SchemaBase', UndefinedType]=Undefined, yError2: Union[dict, 'SchemaBase', UndefinedType]=Undefined, yOffset: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(FacetedEncoding, self).__init__(angle=angle, color=color, column=column, description=description, detail=detail, facet=facet, fill=fill, fillOpacity=fillOpacity, href=href, key=key, latitude=latitude, latitude2=latitude2, longitude=longitude, longitude2=longitude2, opacity=opacity, order=order, radius=radius, radius2=radius2, row=row, shape=shape, size=size, stroke=stroke, strokeDash=strokeDash, strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, text=text, theta=theta, theta2=theta2, tooltip=tooltip, url=url, x=x, x2=x2, xError=xError, xError2=xError2, xOffset=xOffset, y=y, y2=y2, yError=yError, yError2=yError2, yOffset=yOffset, **kwds)