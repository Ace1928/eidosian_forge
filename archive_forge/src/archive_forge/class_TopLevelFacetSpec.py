from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TopLevelFacetSpec(TopLevelSpec):
    """TopLevelFacetSpec schema wrapper

    Parameters
    ----------

    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    facet : dict, :class:`FacetMapping`, :class:`FacetFieldDef`
        Definition for how to facet the data. One of: 1) `a field definition for faceting
        the plot by one field
        <https://vega.github.io/vega-lite/docs/facet.html#field-def>`__ 2) `An object that
        maps row and column channels to their field definitions
        <https://vega.github.io/vega-lite/docs/facet.html#mapping>`__
    spec : dict, :class:`LayerSpec`, :class:`UnitSpecWithFrame`
        A specification of the view that gets faceted.
    align : dict, :class:`LayoutAlign`, :class:`RowColLayoutAlign`, Literal['all', 'each', 'none']
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other.
        * For ``"each"``, subviews will be aligned into a clean grid structure, but each row
          or column may be of variable size.
        * For ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
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
    center : bool, dict, :class:`RowColboolean`
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for:


        * the general (wrappable) ``concat`` operator (not ``hconcat`` / ``vconcat`` )
        * the ``facet`` and ``repeat`` operator with one field/repetition definition
          (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    config : dict, :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
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
    spacing : dict, float, :class:`RowColnumber`
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
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
    _schema = {'$ref': '#/definitions/TopLevelFacetSpec'}

    def __init__(self, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, facet: Union[dict, 'SchemaBase', UndefinedType]=Undefined, spec: Union[dict, 'SchemaBase', UndefinedType]=Undefined, align: Union[dict, 'SchemaBase', Literal['all', 'each', 'none'], UndefinedType]=Undefined, autosize: Union[dict, 'SchemaBase', Literal['pad', 'none', 'fit', 'fit-x', 'fit-y'], UndefinedType]=Undefined, background: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, bounds: Union[Literal['full', 'flush'], UndefinedType]=Undefined, center: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, columns: Union[float, UndefinedType]=Undefined, config: Union[dict, 'SchemaBase', UndefinedType]=Undefined, datasets: Union[dict, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, padding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, params: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, resolve: Union[dict, 'SchemaBase', UndefinedType]=Undefined, spacing: Union[dict, float, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, usermeta: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(TopLevelFacetSpec, self).__init__(data=data, facet=facet, spec=spec, align=align, autosize=autosize, background=background, bounds=bounds, center=center, columns=columns, config=config, datasets=datasets, description=description, name=name, padding=padding, params=params, resolve=resolve, spacing=spacing, title=title, transform=transform, usermeta=usermeta, **kwds)