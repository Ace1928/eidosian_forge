from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class LxmlAttribProtocol(Protocol):
    """A minimal protocol for attribute of lxml Element objects."""

    def get(self, *args: Any, **kwargs: Any) -> Optional[str]:
        ...

    def items(self) -> Sequence[Tuple[Any, Any]]:
        ...

    def __contains__(self, key: Any) -> bool:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __iter__(self) -> Iterator[Any]:
        ...

    def __len__(self) -> int:
        ...