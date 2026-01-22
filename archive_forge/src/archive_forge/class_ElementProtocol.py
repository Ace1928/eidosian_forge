from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class ElementProtocol(Sized, Hashable, Protocol):
    """A protocol for generic ElementTree elements."""

    def __iter__(self) -> Iterator['ElementProtocol']:
        ...

    def find(self, path: str, namespaces: Optional[Dict[str, str]]=...) -> Optional['ElementProtocol']:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator['ElementProtocol']:
        ...

    @overload
    def get(self, key: str) -> Optional[str]:
        ...

    @overload
    def get(self, key: str, default: _T) -> Union[str, _T]:
        ...

    @property
    def tag(self) -> str:
        ...

    @property
    def text(self) -> Optional[str]:
        ...

    @property
    def tail(self) -> Optional[str]:
        ...

    @property
    def attrib(self) -> AttribType:
        ...