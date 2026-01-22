from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Vector12string(VegaLiteSchema):
    """Vector12string schema wrapper"""
    _schema = {'$ref': '#/definitions/Vector12<string>'}

    def __init__(self, *args):
        super(Vector12string, self).__init__(*args)