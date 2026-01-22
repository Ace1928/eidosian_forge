from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class _PathValueError(PydanticValueError):

    def __init__(self, *, path: Path) -> None:
        super().__init__(path=str(path))