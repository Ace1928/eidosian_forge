from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdAttributeGroupProtocol(XsdComponentProtocol, Protocol):

    @overload
    def get(self, key: Optional[str]) -> Optional[XsdAttributeProtocol]:
        ...

    @overload
    def get(self, key: Optional[str], default: _T) -> Union[XsdAttributeProtocol, _T]:
        ...

    def items(self) -> ItemsView[Optional[str], XsdAttributeProtocol]:
        ...

    def __contains__(self, key: Optional[str]) -> bool:
        ...

    def __getitem__(self, key: Optional[str]) -> XsdAttributeProtocol:
        ...

    def __iter__(self) -> Iterator[Optional[str]]:
        ...

    def __len__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    @property
    def ref(self) -> Optional[Any]:
        ...