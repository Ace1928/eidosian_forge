from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class EtreeElementProtocol(ElementProtocol, Protocol):
    """A protocol for xml.etree.ElementTree elements."""

    def __iter__(self) -> Iterator['EtreeElementProtocol']:
        ...

    def find(self, path: str, namespaces: Optional[Dict[str, str]]=...) -> Optional['EtreeElementProtocol']:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator['EtreeElementProtocol']:
        ...

    @property
    def attrib(self) -> Dict[str, str]:
        ...