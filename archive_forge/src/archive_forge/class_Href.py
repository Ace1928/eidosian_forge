import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
@with_property_setters
class Href(FieldChannelMixin, core.StringFieldDefWithCondition):
    """Href schema wrapper

    Parameters
    ----------

    shorthand : str, dict, Sequence[str], :class:`RepeatRef`
        shorthand for field, aggregate, and type
    aggregate : dict, :class:`Aggregate`, :class:`ArgmaxDef`, :class:`ArgminDef`, :class:`NonArgAggregateOp`, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    bandPosition : float
        Relative position on a band of a stacked, binned, time unit, or band scale. For
        example, the marks will be positioned at the beginning of the band if set to ``0``,
        and at the middle of the band if set to ``0.5``.
    bin : str, bool, dict, None, :class:`BinParams`
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#bin-parameters>`__, or indicating
        that the data for ``x`` or ``y`` channel are binned before they are imported into
        Vega-Lite ( ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#bin-parameters>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : dict, :class:`ConditionalValueDefstringExprRef`, :class:`ConditionalParameterValueDefstringExprRef`, :class:`ConditionalPredicateValueDefstringExprRef`, Sequence[dict, :class:`ConditionalValueDefstringExprRef`, :class:`ConditionalParameterValueDefstringExprRef`, :class:`ConditionalPredicateValueDefstringExprRef`]
        One or more value definition(s) with `a parameter or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : str, dict, :class:`Field`, :class:`FieldName`, :class:`RepeatRef`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : str, dict, :class:`Dict`
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__.
        * If the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : str
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:**


        * ``"time"`` for temporal fields and ordinal and nominal fields with ``timeUnit``.
        * ``"number"`` for quantitative fields as well as ordinal and nominal fields without
          ``timeUnit``.
    timeUnit : dict, :class:`TimeUnit`, :class:`MultiTimeUnit`, :class:`BinnedTimeUnit`, :class:`SingleTimeUnit`, :class:`TimeUnitParams`, :class:`UtcMultiTimeUnit`, :class:`UtcSingleTimeUnit`, :class:`LocalMultiTimeUnit`, :class:`LocalSingleTimeUnit`, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds']
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : str, None, :class:`Text`, Sequence[str]
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/usage/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`, Literal['quantitative', 'ordinal', 'temporal', 'nominal']
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria:


        * ``"quantitative"`` is the default type if (1) the encoded field contains ``bin``
          or ``aggregate`` except ``"argmin"`` and ``"argmax"``, (2) the encoding channel is
          ``latitude`` or ``longitude`` channel or (3) if the specified scale type is `a
          quantitative scale <https://vega.github.io/vega-lite/docs/scale.html#type>`__.
        * ``"temporal"`` is the default type if (1) the encoded field contains ``timeUnit``
          or (2) the specified scale type is a time or utc scale
        * ``"ordinal"`` is the default type if (1) the encoded field contains a `custom sort
          order
          <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
          (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
          channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ):


        * ``"quantitative"`` if the datum is a number
        * ``"nominal"`` if the datum is a string
        * ``"temporal"`` if the datum is `a date time object
          <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:**


        * Data ``type`` describes the semantics of the data rather than the primitive data
          types (number, string, etc.). The same primitive data type can have different
          types of measurement. For example, numeric data can represent quantitative,
          ordinal, or nominal data.
        * Data values for a temporal field can be either a date-time string (e.g.,
          ``"2015-03-07 12:32:17"``, ``"17:01"``, ``"2015-03-16"``. ``"2015"`` ) or a
          timestamp number (e.g., ``1552199579097`` ).
        * When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
          ``type`` property can be either ``"quantitative"`` (for using a linear bin scale)
          or `"ordinal" (for using an ordinal bin scale)
          <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__.
        * When using with `timeUnit
          <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type`` property
          can be either ``"temporal"`` (default, for using a temporal scale) or `"ordinal"
          (for using an ordinal scale)
          <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__.
        * When using with `aggregate
          <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type`` property
          refers to the post-aggregation data type. For example, we can calculate count
          ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate": "distinct",
          "field": "cat"}``. The ``"type"`` of the aggregate output is ``"quantitative"``.
        * Secondary channels (e.g., ``x2``, ``y2``, ``xError``, ``yError`` ) do not have
          ``type`` as they must have exactly the same type as their primary channels (e.g.,
          ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _class_is_valid_at_instantiation = False
    _encoding_name = 'href'

    @overload
    def aggregate(self, _: Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], **kwds) -> 'Href':
        ...

    @overload
    def aggregate(self, argmax: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def aggregate(self, argmin: Union[str, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def bandPosition(self, _: float, **kwds) -> 'Href':
        ...

    @overload
    def bin(self, _: bool, **kwds) -> 'Href':
        ...

    @overload
    def bin(self, anchor: Union[float, UndefinedType]=Undefined, base: Union[float, UndefinedType]=Undefined, binned: Union[bool, UndefinedType]=Undefined, divide: Union[Sequence[float], UndefinedType]=Undefined, extent: Union[dict, core._Parameter, core.SchemaBase, Sequence[float], UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, minstep: Union[float, UndefinedType]=Undefined, nice: Union[bool, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, steps: Union[Sequence[float], UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def bin(self, _: str, **kwds) -> 'Href':
        ...

    @overload
    def bin(self, _: None, **kwds) -> 'Href':
        ...

    @overload
    def condition(self, test: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, value: Union[str, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def condition(self, empty: Union[bool, UndefinedType]=Undefined, param: Union[str, core.SchemaBase, UndefinedType]=Undefined, value: Union[str, dict, core._Parameter, core.SchemaBase, UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def condition(self, _: List[core.ConditionalValueDefstringExprRef], **kwds) -> 'Href':
        ...

    @overload
    def field(self, _: str, **kwds) -> 'Href':
        ...

    @overload
    def field(self, repeat: Union[Literal['row', 'column', 'repeat', 'layer'], UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def format(self, _: str, **kwds) -> 'Href':
        ...

    @overload
    def format(self, _: dict, **kwds) -> 'Href':
        ...

    @overload
    def formatType(self, _: str, **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, _: Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], **kwds) -> 'Href':
        ...

    @overload
    def timeUnit(self, binned: Union[bool, UndefinedType]=Undefined, maxbins: Union[float, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, unit: Union[core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, utc: Union[bool, UndefinedType]=Undefined, **kwds) -> 'Href':
        ...

    @overload
    def title(self, _: str, **kwds) -> 'Href':
        ...

    @overload
    def title(self, _: List[str], **kwds) -> 'Href':
        ...

    @overload
    def title(self, _: None, **kwds) -> 'Href':
        ...

    @overload
    def type(self, _: Literal['quantitative', 'ordinal', 'temporal', 'nominal'], **kwds) -> 'Href':
        ...

    def __init__(self, shorthand: Union[str, dict, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, aggregate: Union[dict, core.SchemaBase, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, bandPosition: Union[float, UndefinedType]=Undefined, bin: Union[str, bool, dict, None, core.SchemaBase, UndefinedType]=Undefined, condition: Union[dict, core.SchemaBase, Sequence[Union[dict, core.SchemaBase]], UndefinedType]=Undefined, field: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, format: Union[str, dict, core.SchemaBase, UndefinedType]=Undefined, formatType: Union[str, UndefinedType]=Undefined, timeUnit: Union[dict, core.SchemaBase, Literal['year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes', 'seconds', 'milliseconds'], Literal['utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate', 'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds'], Literal['binnedyear', 'binnedyearquarter', 'binnedyearquartermonth', 'binnedyearmonth', 'binnedyearmonthdate', 'binnedyearmonthdatehours', 'binnedyearmonthdatehoursminutes', 'binnedyearmonthdatehoursminutesseconds', 'binnedyearweek', 'binnedyearweekday', 'binnedyearweekdayhours', 'binnedyearweekdayhoursminutes', 'binnedyearweekdayhoursminutesseconds', 'binnedyeardayofyear'], Literal['binnedutcyear', 'binnedutcyearquarter', 'binnedutcyearquartermonth', 'binnedutcyearmonth', 'binnedutcyearmonthdate', 'binnedutcyearmonthdatehours', 'binnedutcyearmonthdatehoursminutes', 'binnedutcyearmonthdatehoursminutesseconds', 'binnedutcyearweek', 'binnedutcyearweekday', 'binnedutcyearweekdayhours', 'binnedutcyearweekdayhoursminutes', 'binnedutcyearweekdayhoursminutesseconds', 'binnedutcyeardayofyear'], Literal['yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours', 'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday', 'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds', 'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes', 'monthdatehoursminutesseconds', 'weekday', 'weekdayhours', 'weekdayhoursminutes', 'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds', 'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds'], Literal['utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate', 'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes', 'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday', 'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds', 'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours', 'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday', 'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds', 'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes', 'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds'], UndefinedType]=Undefined, title: Union[str, None, Sequence[str], core.SchemaBase, UndefinedType]=Undefined, type: Union[core.SchemaBase, Literal['quantitative', 'ordinal', 'temporal', 'nominal'], UndefinedType]=Undefined, **kwds):
        super(Href, self).__init__(shorthand=shorthand, aggregate=aggregate, bandPosition=bandPosition, bin=bin, condition=condition, field=field, format=format, formatType=formatType, timeUnit=timeUnit, title=title, type=type, **kwds)