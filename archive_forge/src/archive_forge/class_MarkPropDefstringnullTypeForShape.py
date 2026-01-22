from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class MarkPropDefstringnullTypeForShape(VegaLiteSchema):
    """MarkPropDefstringnullTypeForShape schema wrapper"""
    _schema = {'$ref': '#/definitions/MarkPropDef<(string|null),TypeForShape>'}

    def __init__(self, *args, **kwds):
        super(MarkPropDefstringnullTypeForShape, self).__init__(*args, **kwds)