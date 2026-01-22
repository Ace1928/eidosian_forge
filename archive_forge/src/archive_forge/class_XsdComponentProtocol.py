from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdComponentProtocol(XsdValidatorProtocol, Protocol):

    @property
    def parent(self) -> Optional['XsdComponentProtocol']:
        ...