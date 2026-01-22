from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ArgminDef(Aggregate):
    """ArgminDef schema wrapper

    Parameters
    ----------

    argmin : str, :class:`FieldName`

    """
    _schema = {'$ref': '#/definitions/ArgminDef'}

    def __init__(self, argmin: Union[str, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(ArgminDef, self).__init__(argmin=argmin, **kwds)