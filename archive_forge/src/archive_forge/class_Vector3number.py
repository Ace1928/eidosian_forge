from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Vector3number(VegaLiteSchema):
    """Vector3number schema wrapper"""
    _schema = {'$ref': '#/definitions/Vector3<number>'}

    def __init__(self, *args):
        super(Vector3number, self).__init__(*args)