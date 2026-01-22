from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class NonNormalizedSpec(VegaLiteSchema):
    """NonNormalizedSpec schema wrapper
    Any specification in Vega-Lite.
    """
    _schema = {'$ref': '#/definitions/NonNormalizedSpec'}

    def __init__(self, *args, **kwds):
        super(NonNormalizedSpec, self).__init__(*args, **kwds)