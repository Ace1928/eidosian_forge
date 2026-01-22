from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Vector2string(SelectionInitInterval):
    """Vector2string schema wrapper"""
    _schema = {'$ref': '#/definitions/Vector2<string>'}

    def __init__(self, *args):
        super(Vector2string, self).__init__(*args)