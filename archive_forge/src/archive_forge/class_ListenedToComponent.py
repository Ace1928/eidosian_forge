from typing import (
from thinc.api import Model, Optimizer
from .compat import Protocol, runtime_checkable
@runtime_checkable
class ListenedToComponent(Protocol):
    model: Any
    listeners: Sequence[Model]
    listener_map: Dict[str, Sequence[Model]]
    listening_components: List[str]

    def add_listener(self, listener: Model, component_name: str) -> None:
        ...

    def remove_listener(self, listener: Model, component_name: str) -> bool:
        ...

    def find_listeners(self, component) -> None:
        ...