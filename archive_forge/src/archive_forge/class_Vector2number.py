from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Vector2number(SelectionInitInterval):
    """Vector2number schema wrapper"""
    _schema = {'$ref': '#/definitions/Vector2<number>'}

    def __init__(self, *args):
        super(Vector2number, self).__init__(*args)