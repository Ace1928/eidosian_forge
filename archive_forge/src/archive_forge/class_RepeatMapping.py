from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class RepeatMapping(VegaLiteSchema):
    """RepeatMapping schema wrapper

    Parameters
    ----------

    column : Sequence[str]
        An array of fields to be repeated horizontally.
    row : Sequence[str]
        An array of fields to be repeated vertically.
    """
    _schema = {'$ref': '#/definitions/RepeatMapping'}

    def __init__(self, column: Union[Sequence[str], UndefinedType]=Undefined, row: Union[Sequence[str], UndefinedType]=Undefined, **kwds):
        super(RepeatMapping, self).__init__(column=column, row=row, **kwds)