import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class InternalPlaceholderContext(trace.PlaceholderContext):
    """Container with mappings shared across TraceTypes for placeholder values."""

    def __init__(self, context_graph=None, placeholder_mapping=None, unnest_only=False, with_none_control_dependencies=False, composite_device_name=None):
        self._alias_id_to_placeholder = placeholder_mapping or {}
        self._naming_scope = None
        self._context_graph = context_graph
        self._unnest_only = unnest_only
        self._with_none_control_dependencies = with_none_control_dependencies
        self._composite_device_name = composite_device_name

    def has_placeholder(self, alias_id: Hashable) -> bool:
        return alias_id in self._alias_id_to_placeholder

    def get_placeholder(self, alias_id: Hashable) -> Hashable:
        if not self.has_placeholder(alias_id):
            raise KeyError(f'alias_id: {alias_id} not found in this instance of placeholder context.')
        return self._alias_id_to_placeholder[alias_id]

    def add_placeholder(self, alias_id: Hashable, placeholder: Hashable) -> None:
        if alias_id in self._alias_id_to_placeholder:
            raise KeyError(f'alias id: {alias_id} is already stored in this instance of placeholder context.')
        self._alias_id_to_placeholder[alias_id] = placeholder

    def update_naming_scope(self, naming_scope: Optional[str]) -> None:
        self._naming_scope = naming_scope

    @property
    def naming_scope(self) -> Optional[str]:
        return self._naming_scope

    @property
    def context_graph(self):
        return self._context_graph

    @property
    def unnest_only(self) -> bool:
        return self._unnest_only

    @property
    def with_none_control_dependencies(self) -> bool:
        return self._with_none_control_dependencies

    @property
    def composite_device_name(self) -> Any:
        return self._composite_device_name