from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class _Parameter(Protocol):
    _counter: int

    def _get_name(cls) -> str:
        ...

    def to_dict(self) -> TypingDict[str, Union[str, dict]]:
        ...

    def _to_expr(self) -> str:
        ...