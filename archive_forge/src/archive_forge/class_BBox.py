from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BBox(VegaLiteSchema):
    """BBox schema wrapper
    Bounding box https://tools.ietf.org/html/rfc7946#section-5
    """
    _schema = {'$ref': '#/definitions/BBox'}

    def __init__(self, *args, **kwds):
        super(BBox, self).__init__(*args, **kwds)