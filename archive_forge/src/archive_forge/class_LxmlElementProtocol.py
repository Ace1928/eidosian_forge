from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class LxmlElementProtocol(ElementProtocol, Protocol):
    """A protocol for lxml.etree elements."""

    def __iter__(self) -> Iterator['LxmlElementProtocol']:
        ...

    def find(self, path: str, namespaces: Optional[Dict[str, str]]=...) -> Optional['LxmlElementProtocol']:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator['LxmlElementProtocol']:
        ...

    def getroottree(self) -> 'LxmlDocumentProtocol':
        ...

    def getnext(self) -> Optional['LxmlElementProtocol']:
        ...

    def getparent(self) -> Optional['LxmlElementProtocol']:
        ...

    def getprevious(self) -> Optional['LxmlElementProtocol']:
        ...

    def itersiblings(self, tag: Optional[str]=..., *tags: str, preceding: bool=False) -> Iterable['LxmlElementProtocol']:
        ...

    @property
    def nsmap(self) -> Dict[Optional[str], str]:
        ...

    @property
    def attrib(self) -> LxmlAttribProtocol:
        ...