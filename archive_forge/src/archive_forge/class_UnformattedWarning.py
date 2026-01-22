import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
@final
@dataclasses.dataclass
class UnformattedWarning(Generic[_W]):
    """A warning meant to be formatted during runtime.

    This is used to hold warnings that need to format their message at runtime,
    as opposed to a direct message.
    """
    category: Type['_W']
    template: str

    def format(self, **kwargs: Any) -> _W:
        """Return an instance of the warning category, formatted with given kwargs."""
        return self.category(self.template.format(**kwargs))