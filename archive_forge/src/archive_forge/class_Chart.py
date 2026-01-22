import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
class Chart(TopLevelMixin, _EncodingMixin, mixins.MarkMethodMixin, core.TopLevelUnitSpec):
    """Create a basic Altair/Vega-Lite chart.

    Although it is possible to set all Chart properties as constructor attributes,
    it is more idiomatic to use methods such as ``mark_point()``, ``encode()``,
    ``transform_filter()``, ``properties()``, etc. See Altair's documentation
    for details and examples: http://altair-viz.github.io/.

    Parameters
    ----------
    data : Data
        An object describing the data source
    mark : AnyMark
        A `MarkDef` or `CompositeMarkDef` object, or a string describing the mark type
        (one of `"arc"`, `"area"`, `"bar"`, `"circle"`, `"geoshape"`, `"image"`,
        `"line"`, `"point"`, `"rule"`, `"rect"`, `"square"`, `"text"`, `"tick"`,
        `"trail"`, `"boxplot"`, `"errorband"`, and `"errorbar"`).
    encoding : FacetedEncoding
        A key-value mapping between encoding channels and definition of fields.
    autosize : anyOf(AutosizeType, AutoSizeParams)
        Sets how the visualization size should be determined. If a string, should be one of
        `"pad"`, `"fit"` or `"none"`. Object values can additionally specify parameters for
        content sizing and automatic resizing. `"fit"` is only supported for single and
        layered views that don't use `rangeStep`.  Default value: `pad`
    background : string
        CSS color property to use as the background of visualization.

        **Default value:** none (transparent)
    config : Config
        Vega-Lite configuration object.  This property can only be defined at the top-level
        of a specification.
    description : string
        Description of this mark for commenting purpose.
    height : float
        The height of a visualization.
    name : string
        Name of the visualization for later reference.
    padding : Padding
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle.  If a number, specifies padding for all sides. If an
        object, the value should have the format `{"left": 5, "top": 5, "right": 5,
        "bottom": 5}` to specify padding for each side of the visualization.  Default
        value: `5`
    projection : Projection
        An object defining properties of geographic projection.  Works with `"geoshape"`
        marks and `"point"` or `"line"` marks that have a channel (one or more of `"X"`,
        `"X2"`, `"Y"`, `"Y2"`) with type `"latitude"`, or `"longitude"`.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    title : anyOf(string, TitleParams)
        Title for the plot.
    transform : List(Transform)
        An array of data transformations such as filter and new field calculation.
    width : float
        The width of a visualization.
    """

    def __init__(self, data: Union[ChartDataType, UndefinedType]=Undefined, encoding: Union[core.FacetedEncoding, UndefinedType]=Undefined, mark: Union[str, core.AnyMark, UndefinedType]=Undefined, width: Union[int, str, dict, core.Step, UndefinedType]=Undefined, height: Union[int, str, dict, core.Step, UndefinedType]=Undefined, **kwargs) -> None:
        super(Chart, self).__init__(data=data, encoding=encoding, mark=mark, width=width, height=height, **kwargs)
    _counter: int = 0

    @classmethod
    def _get_name(cls) -> str:
        cls._counter += 1
        return f'view_{cls._counter}'

    @classmethod
    def from_dict(cls, dct: dict, validate: bool=True) -> core.SchemaBase:
        """Construct class from a dictionary representation

        Parameters
        ----------
        dct : dictionary
            The dict from which to construct the class
        validate : boolean
            If True (default), then validate the input against the schema.

        Returns
        -------
        obj : Chart object
            The wrapped schema

        Raises
        ------
        jsonschema.ValidationError :
            if validate=True and dct does not conform to the schema
        """
        for class_ in TopLevelMixin.__subclasses__():
            if class_ is Chart:
                class_ = cast(TypingType[TopLevelMixin], super(Chart, cls))
            try:
                return class_.from_dict(dct, validate=validate)
            except jsonschema.ValidationError:
                pass
        return core.Root.from_dict(dct, validate)

    def to_dict(self, validate: bool=True, *, format: str='vega-lite', ignore: Optional[List[str]]=None, context: Optional[TypingDict[str, Any]]=None) -> dict:
        """Convert the chart to a dictionary suitable for JSON export

        Parameters
        ----------
        validate : bool, optional
            If True (default), then validate the output dictionary
            against the schema.
        format : str, optional
            Chart specification format, one of "vega-lite" (default) or "vega"
        ignore : list[str], optional
            A list of keys to ignore. It is usually not needed
            to specify this argument as a user.
        context : dict[str, Any], optional
            A context dictionary. It is usually not needed
            to specify this argument as a user.

        Notes
        -----
        Technical: The ignore parameter will *not* be passed to child to_dict
        function calls.

        Returns
        -------
        dict
            The dictionary representation of this chart

        Raises
        ------
        SchemaValidationError
            if validate=True and the dict does not conform to the schema
        """
        context = context or {}
        if self.data is Undefined and 'data' not in context:
            copy = self.copy(deep=False)
            copy.data = core.InlineData(values=[{}])
            return super(Chart, copy).to_dict(validate=validate, format=format, ignore=ignore, context=context)
        return super().to_dict(validate=validate, format=format, ignore=ignore, context=context)

    def transformed_data(self, row_limit: Optional[int]=None, exclude: Optional[Iterable[str]]=None) -> Optional[DataFrameLike]:
        """Evaluate a Chart's transforms

        Evaluate the data transforms associated with a Chart and return the
        transformed data a DataFrame

        Parameters
        ----------
        row_limit : int (optional)
            Maximum number of rows to return for each DataFrame. None (default) for unlimited
        exclude : iterable of str
            Set of the names of charts to exclude

        Returns
        -------
        DataFrame
            Transformed data as a DataFrame
        """
        from altair.utils._transformed_data import transformed_data
        return transformed_data(self, row_limit=row_limit, exclude=exclude)

    def add_params(self, *params: Parameter) -> Self:
        """Add one or more parameters to the chart."""
        if not params:
            return self
        copy = self.copy(deep=['params'])
        if copy.params is Undefined:
            copy.params = []
        for s in params:
            copy.params.append(s.param)
        return copy

    @utils.deprecation.deprecated(message="'add_selection' is deprecated. Use 'add_params' instead.")
    def add_selection(self, *params) -> Self:
        """'add_selection' is deprecated. Use 'add_params' instead."""
        return self.add_params(*params)

    def interactive(self, name: Optional[str]=None, bind_x: bool=True, bind_y: bool=True) -> Self:
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The parameter name to use for the axes scales. This name should be
            unique among all parameters within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        encodings = []
        if bind_x:
            encodings.append('x')
        if bind_y:
            encodings.append('y')
        return self.add_params(selection_interval(bind='scales', encodings=encodings))