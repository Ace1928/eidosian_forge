from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LegendStreamBinding(LegendBinding):
    """LegendStreamBinding schema wrapper

    Parameters
    ----------

    legend : str, dict, :class:`Stream`, :class:`EventStream`, :class:`MergedStream`, :class:`DerivedStream`

    """
    _schema = {'$ref': '#/definitions/LegendStreamBinding'}

    def __init__(self, legend: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(LegendStreamBinding, self).__init__(legend=legend, **kwds)