from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Cyclical(ColorScheme):
    """Cyclical schema wrapper"""
    _schema = {'$ref': '#/definitions/Cyclical'}

    def __init__(self, *args):
        super(Cyclical, self).__init__(*args)