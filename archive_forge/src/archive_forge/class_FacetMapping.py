from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FacetMapping(VegaLiteSchema):
    """FacetMapping schema wrapper

    Parameters
    ----------

    column : dict, :class:`FacetFieldDef`
        A field definition for the horizontal facet of trellis plots.
    row : dict, :class:`FacetFieldDef`
        A field definition for the vertical facet of trellis plots.
    """
    _schema = {'$ref': '#/definitions/FacetMapping'}

    def __init__(self, column: Union[dict, 'SchemaBase', UndefinedType]=Undefined, row: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(FacetMapping, self).__init__(column=column, row=row, **kwds)