from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdAttributeProtocol(XsdComponentProtocol, Protocol):

    @property
    def type(self) -> Optional[XsdTypeProtocol]:
        ...

    @property
    def ref(self) -> Optional[Any]:
        ...