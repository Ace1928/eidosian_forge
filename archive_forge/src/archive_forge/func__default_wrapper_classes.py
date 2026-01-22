from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
@classmethod
def _default_wrapper_classes(cls) -> TypingGenerator[type, None, None]:
    return _subclasses(VegaLiteSchema)