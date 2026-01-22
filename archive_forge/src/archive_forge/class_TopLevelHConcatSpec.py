from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TopLevelHConcatSpec(TopLevelSpec):
    """TopLevelHConcatSpec schema wrapper

    Parameters
    ----------

    hconcat : Sequence[dict, :class:`FacetSpec`, :class:`LayerSpec`, :class:`RepeatSpec`, :class:`FacetedUnitSpec`, :class:`LayerRepeatSpec`, :class:`NonNormalizedSpec`, :class:`NonLayerRepeatSpec`, :class:`ConcatSpecGenericSpec`, :class:`HConcatSpecGenericSpec`, :class:`VConcatSpecGenericSpec`]
        A list of views to be concatenated and put into a row.
    autosize : dict, :class:`AutosizeType`, :class:`AutoSizeParams`, Literal['pad', 'none', 'fit', 'fit-x', 'fit-y']
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : str, dict, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : Literal['full', 'flush']
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used.
        * If set to ``flush``, only the specified width and height values for the sub-view
          will be used. The ``flush`` setting can be useful when attempting to place
          sub-plots without axes or legends into a uniform grid structure.

        **Default value:** ``"full"``
    center : bool
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    config : dict, :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    datasets : dict, :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : str
        Description of this mark for commenting purpose.
    name : str
        Name of the visualization for later reference.
    padding : dict, float, :class:`ExprRef`, :class:`Padding`
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : Sequence[dict, :class:`TopLevelParameter`, :class:`VariableParameter`, :class:`TopLevelSelectionParameter`]
        Dynamic variables or selections that parameterize a visualization.
    resolve : dict, :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : str, dict, :class:`Text`, Sequence[str], :class:`TitleParams`
        Title for the plot.
    transform : Sequence[dict, :class:`Transform`, :class:`BinTransform`, :class:`FoldTransform`, :class:`LoessTransform`, :class:`PivotTransform`, :class:`StackTransform`, :class:`ExtentTransform`, :class:`FilterTransform`, :class:`ImputeTransform`, :class:`LookupTransform`, :class:`SampleTransform`, :class:`WindowTransform`, :class:`DensityTransform`, :class:`FlattenTransform`, :class:`QuantileTransform`, :class:`TimeUnitTransform`, :class:`AggregateTransform`, :class:`CalculateTransform`, :class:`RegressionTransform`, :class:`JoinAggregateTransform`]
        An array of data transformations such as filter and new field calculation.
    usermeta : dict, :class:`Dict`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    $schema : str
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v5.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelHConcatSpec'}

    def __init__(self, hconcat: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, autosize: Union[dict, 'SchemaBase', Literal['pad', 'none', 'fit', 'fit-x', 'fit-y'], UndefinedType]=Undefined, background: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, bounds: Union[Literal['full', 'flush'], UndefinedType]=Undefined, center: Union[bool, UndefinedType]=Undefined, config: Union[dict, 'SchemaBase', UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, datasets: Union[dict, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, padding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, params: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, resolve: Union[dict, 'SchemaBase', UndefinedType]=Undefined, spacing: Union[float, UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, usermeta: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(TopLevelHConcatSpec, self).__init__(hconcat=hconcat, autosize=autosize, background=background, bounds=bounds, center=center, config=config, data=data, datasets=datasets, description=description, name=name, padding=padding, params=params, resolve=resolve, spacing=spacing, title=title, transform=transform, usermeta=usermeta, **kwds)