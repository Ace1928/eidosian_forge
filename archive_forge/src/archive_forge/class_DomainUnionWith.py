from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DomainUnionWith(VegaLiteSchema):
    """DomainUnionWith schema wrapper

    Parameters
    ----------

    unionWith : Sequence[str, bool, dict, float, :class:`DateTime`]
        Customized domain values to be union with the field's values or explicitly defined
        domain. Should be an array of valid scale domain values.
    """
    _schema = {'$ref': '#/definitions/DomainUnionWith'}

    def __init__(self, unionWith: Union[Sequence[Union[str, bool, dict, float, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(DomainUnionWith, self).__init__(unionWith=unionWith, **kwds)