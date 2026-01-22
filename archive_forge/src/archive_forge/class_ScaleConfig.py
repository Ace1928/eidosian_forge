from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ScaleConfig(VegaLiteSchema):
    """ScaleConfig schema wrapper

    Parameters
    ----------

    bandPaddingInner : dict, float, :class:`ExprRef`
        Default inner padding for ``x`` and ``y`` band scales.

        **Default value:**


        * ``nestedOffsetPaddingInner`` for x/y scales with nested x/y offset scales.
        * ``barBandPaddingInner`` for bar marks ( ``0.1`` by default)
        * ``rectBandPaddingInner`` for rect and other marks ( ``0`` by default)
    bandPaddingOuter : dict, float, :class:`ExprRef`
        Default outer padding for ``x`` and ``y`` band scales.

        **Default value:** ``paddingInner/2`` (which makes *width/height = number of unique
        values * step* )
    bandWithNestedOffsetPaddingInner : dict, float, :class:`ExprRef`
        Default inner padding for ``x`` and ``y`` band scales with nested ``xOffset`` and
        ``yOffset`` encoding.

        **Default value:** ``0.2``
    bandWithNestedOffsetPaddingOuter : dict, float, :class:`ExprRef`
        Default outer padding for ``x`` and ``y`` band scales with nested ``xOffset`` and
        ``yOffset`` encoding.

        **Default value:** ``0.2``
    barBandPaddingInner : dict, float, :class:`ExprRef`
        Default inner padding for ``x`` and ``y`` band-ordinal scales of ``"bar"`` marks.

        **Default value:** ``0.1``
    clamp : bool, dict, :class:`ExprRef`
        If true, values that exceed the data domain are clamped to either the minimum or
        maximum range value
    continuousPadding : dict, float, :class:`ExprRef`
        Default padding for continuous x/y scales.

        **Default:** The bar width for continuous x-scale of a vertical bar and continuous
        y-scale of a horizontal bar.; ``0`` otherwise.
    maxBandSize : float
        The default max value for mapping quantitative fields to bar's size/bandSize.

        If undefined (default), we will use the axis's size (width or height) - 1.
    maxFontSize : float
        The default max value for mapping quantitative fields to text's size/fontSize.

        **Default value:** ``40``
    maxOpacity : float
        Default max opacity for mapping a field to opacity.

        **Default value:** ``0.8``
    maxSize : float
        Default max value for point size scale.
    maxStrokeWidth : float
        Default max strokeWidth for the scale of strokeWidth for rule and line marks and of
        size for trail marks.

        **Default value:** ``4``
    minBandSize : float
        The default min value for mapping quantitative fields to bar and tick's
        size/bandSize scale with zero=false.

        **Default value:** ``2``
    minFontSize : float
        The default min value for mapping quantitative fields to tick's size/fontSize scale
        with zero=false

        **Default value:** ``8``
    minOpacity : float
        Default minimum opacity for mapping a field to opacity.

        **Default value:** ``0.3``
    minSize : float
        Default minimum value for point size scale with zero=false.

        **Default value:** ``9``
    minStrokeWidth : float
        Default minimum strokeWidth for the scale of strokeWidth for rule and line marks and
        of size for trail marks with zero=false.

        **Default value:** ``1``
    offsetBandPaddingInner : dict, float, :class:`ExprRef`
        Default padding inner for xOffset/yOffset's band scales.

        **Default Value:** ``0``
    offsetBandPaddingOuter : dict, float, :class:`ExprRef`
        Default padding outer for xOffset/yOffset's band scales.

        **Default Value:** ``0``
    pointPadding : dict, float, :class:`ExprRef`
        Default outer padding for ``x`` and ``y`` point-ordinal scales.

        **Default value:** ``0.5`` (which makes *width/height = number of unique values *
        step* )
    quantileCount : float
        Default range cardinality for `quantile
        <https://vega.github.io/vega-lite/docs/scale.html#quantile>`__ scale.

        **Default value:** ``4``
    quantizeCount : float
        Default range cardinality for `quantize
        <https://vega.github.io/vega-lite/docs/scale.html#quantize>`__ scale.

        **Default value:** ``4``
    rectBandPaddingInner : dict, float, :class:`ExprRef`
        Default inner padding for ``x`` and ``y`` band-ordinal scales of ``"rect"`` marks.

        **Default value:** ``0``
    round : bool, dict, :class:`ExprRef`
        If true, rounds numeric output values to integers. This can be helpful for snapping
        to the pixel grid. (Only available for ``x``, ``y``, and ``size`` scales.)
    useUnaggregatedDomain : bool
        Use the source data range before aggregation as scale domain instead of aggregated
        data for aggregate axis.

        This is equivalent to setting ``domain`` to ``"unaggregate"`` for aggregated
        *quantitative* fields by default.

        This property only works with aggregate functions that produce values within the raw
        data domain ( ``"mean"``, ``"average"``, ``"median"``, ``"q1"``, ``"q3"``,
        ``"min"``, ``"max"`` ). For other aggregations that produce values outside of the
        raw data domain (e.g. ``"count"``, ``"sum"`` ), this property is ignored.

        **Default value:** ``false``
    xReverse : bool, dict, :class:`ExprRef`
        Reverse x-scale by default (useful for right-to-left charts).
    zero : bool
        Default ``scale.zero`` for `continuous
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ scales except for
        (1) x/y-scales of non-ranged bar or area charts and (2) size scales.

        **Default value:** ``true``
    """
    _schema = {'$ref': '#/definitions/ScaleConfig'}

    def __init__(self, bandPaddingInner: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, bandPaddingOuter: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, bandWithNestedOffsetPaddingInner: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, bandWithNestedOffsetPaddingOuter: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, barBandPaddingInner: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, clamp: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, continuousPadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, maxBandSize: Union[float, UndefinedType]=Undefined, maxFontSize: Union[float, UndefinedType]=Undefined, maxOpacity: Union[float, UndefinedType]=Undefined, maxSize: Union[float, UndefinedType]=Undefined, maxStrokeWidth: Union[float, UndefinedType]=Undefined, minBandSize: Union[float, UndefinedType]=Undefined, minFontSize: Union[float, UndefinedType]=Undefined, minOpacity: Union[float, UndefinedType]=Undefined, minSize: Union[float, UndefinedType]=Undefined, minStrokeWidth: Union[float, UndefinedType]=Undefined, offsetBandPaddingInner: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, offsetBandPaddingOuter: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, pointPadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, quantileCount: Union[float, UndefinedType]=Undefined, quantizeCount: Union[float, UndefinedType]=Undefined, rectBandPaddingInner: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, round: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, useUnaggregatedDomain: Union[bool, UndefinedType]=Undefined, xReverse: Union[bool, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, zero: Union[bool, UndefinedType]=Undefined, **kwds):
        super(ScaleConfig, self).__init__(bandPaddingInner=bandPaddingInner, bandPaddingOuter=bandPaddingOuter, bandWithNestedOffsetPaddingInner=bandWithNestedOffsetPaddingInner, bandWithNestedOffsetPaddingOuter=bandWithNestedOffsetPaddingOuter, barBandPaddingInner=barBandPaddingInner, clamp=clamp, continuousPadding=continuousPadding, maxBandSize=maxBandSize, maxFontSize=maxFontSize, maxOpacity=maxOpacity, maxSize=maxSize, maxStrokeWidth=maxStrokeWidth, minBandSize=minBandSize, minFontSize=minFontSize, minOpacity=minOpacity, minSize=minSize, minStrokeWidth=minStrokeWidth, offsetBandPaddingInner=offsetBandPaddingInner, offsetBandPaddingOuter=offsetBandPaddingOuter, pointPadding=pointPadding, quantileCount=quantileCount, quantizeCount=quantizeCount, rectBandPaddingInner=rectBandPaddingInner, round=round, useUnaggregatedDomain=useUnaggregatedDomain, xReverse=xReverse, zero=zero, **kwds)