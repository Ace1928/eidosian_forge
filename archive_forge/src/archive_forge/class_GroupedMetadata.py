import math
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union
@runtime_checkable
class GroupedMetadata(Protocol):
    """A grouping of multiple BaseMetadata objects.

    `GroupedMetadata` on its own is not metadata and has no meaning.
    All it the the constraint and metadata should be fully expressable
    in terms of the `BaseMetadata`'s returned by `GroupedMetadata.__iter__()`.

    Concrete implementations should override `GroupedMetadata.__iter__()`
    to add their own metadata.
    For example:

    >>> @dataclass
    >>> class Field(GroupedMetadata):
    >>>     gt: float | None = None
    >>>     description: str | None = None
    ...
    >>>     def __iter__(self) -> Iterable[BaseMetadata]:
    >>>         if self.gt is not None:
    >>>             yield Gt(self.gt)
    >>>         if self.description is not None:
    >>>             yield Description(self.gt)

    Also see the implementation of `Interval` below for an example.

    Parsers should recognize this and unpack it so that it can be used
    both with and without unpacking:

    - `Annotated[int, Field(...)]` (parser must unpack Field)
    - `Annotated[int, *Field(...)]` (PEP-646)
    """

    @property
    def __is_annotated_types_grouped_metadata__(self) -> Literal[True]:
        return True

    def __iter__(self) -> Iterator[BaseMetadata]:
        ...
    if not TYPE_CHECKING:
        __slots__ = ()

        def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
            super().__init_subclass__(*args, **kwargs)
            if cls.__iter__ is GroupedMetadata.__iter__:
                raise TypeError("Can't subclass GroupedMetadata without implementing __iter__")

        def __iter__(self) -> Iterator[BaseMetadata]:
            raise NotImplementedError