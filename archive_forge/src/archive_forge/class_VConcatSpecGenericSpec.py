from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class VConcatSpecGenericSpec(Spec, NonNormalizedSpec):
    """VConcatSpecGenericSpec schema wrapper
    Base interface for a vertical concatenation specification.

    Parameters
    ----------

    vconcat : Sequence[dict, :class:`Spec`, :class:`FacetSpec`, :class:`LayerSpec`, :class:`RepeatSpec`, :class:`FacetedUnitSpec`, :class:`LayerRepeatSpec`, :class:`NonLayerRepeatSpec`, :class:`ConcatSpecGenericSpec`, :class:`HConcatSpecGenericSpec`, :class:`VConcatSpecGenericSpec`]
        A list of views to be concatenated and put into a column.
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
    data : dict, None, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : str
        Description of this mark for commenting purpose.
    name : str
        Name of the visualization for later reference.
    resolve : dict, :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : str, dict, :class:`Text`, Sequence[str], :class:`TitleParams`
        Title for the plot.
    transform : Sequence[dict, :class:`Transform`, :class:`BinTransform`, :class:`FoldTransform`, :class:`LoessTransform`, :class:`PivotTransform`, :class:`StackTransform`, :class:`ExtentTransform`, :class:`FilterTransform`, :class:`ImputeTransform`, :class:`LookupTransform`, :class:`SampleTransform`, :class:`WindowTransform`, :class:`DensityTransform`, :class:`FlattenTransform`, :class:`QuantileTransform`, :class:`TimeUnitTransform`, :class:`AggregateTransform`, :class:`CalculateTransform`, :class:`RegressionTransform`, :class:`JoinAggregateTransform`]
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/VConcatSpec<GenericSpec>'}

    def __init__(self, vconcat: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, bounds: Union[Literal['full', 'flush'], UndefinedType]=Undefined, center: Union[bool, UndefinedType]=Undefined, data: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, description: Union[str, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, resolve: Union[dict, 'SchemaBase', UndefinedType]=Undefined, spacing: Union[float, UndefinedType]=Undefined, title: Union[str, dict, 'SchemaBase', Sequence[str], UndefinedType]=Undefined, transform: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(VConcatSpecGenericSpec, self).__init__(vconcat=vconcat, bounds=bounds, center=center, data=data, description=description, name=name, resolve=resolve, spacing=spacing, title=title, transform=transform, **kwds)