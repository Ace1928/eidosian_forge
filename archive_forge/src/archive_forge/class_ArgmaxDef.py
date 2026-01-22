from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ArgmaxDef(Aggregate):
    """ArgmaxDef schema wrapper

    Parameters
    ----------

    argmax : str, :class:`FieldName`

    """
    _schema = {'$ref': '#/definitions/ArgmaxDef'}

    def __init__(self, argmax: Union[str, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(ArgmaxDef, self).__init__(argmax=argmax, **kwds)