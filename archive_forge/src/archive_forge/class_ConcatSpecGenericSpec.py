from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ConcatSpecGenericSpec(Spec, NonNormalizedSpec):
    """ConcatSpecGenericSpec schema wrapper
    Base interface for a generalized concatenation specification.

    Parameters
    ----------

    concat : Sequence[dict, :class:`Spec`, :class:`FacetSpec`, :class:`LayerSpec`, :class:`RepeatSpec`, :class:`FacetedUnitSpec`, :class:`LayerRepeatSpec`, :class:`NonLayerRepeatSpec`, :class:`ConcatSpecGenericSpec`, :class:`HConcatSpecGenericSpec`, :class:`VConcatSpecGenericSpec`]
        A list of views to be concatenated.
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
    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : str
        Description of this mark for commenting purpose.
    name : str
        Name of the visualization for later reference.
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
    """
    _schema = {'$ref': '#/definitions/ConcatSpec<GenericSpec>'}

    def __init__(self, concat: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, align: Union[dict, 'SchemaBase', Literal['all', 'each', 'none'], UndefinedType]=Undefined, bounds: Union[Literal['full', 'flush'], UndefinedType]=Undefined, center: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, columns: Union[float, UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, resolve: Union[dict, 'SchemaBase', UndefinedType]=Undefined, spacing: Union[dict, float, 'SchemaBase', UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(ConcatSpecGenericSpec, self).__init__(concat=concat, align=align, bounds=bounds, center=center, columns=columns, data=data, description=description, name=name, resolve=resolve, spacing=spacing, title=title, transform=transform, **kwds)