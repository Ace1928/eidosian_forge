import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union, overload
from .._fields import MISSING_NONPROP
@dataclasses.dataclass(frozen=True)
class _ArgConfiguration:
    name: Optional[str]
    metavar: Optional[str]
    help: Optional[str]
    aliases: Optional[Tuple[str, ...]]
    prefix_name: Optional[bool]
    constructor_factory: Optional[Callable[[], Union[Type, Callable]]]