from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def update_component_schema_registry(self, components: Dict[str, Union[str, Dict]], kind: Optional[str]=None, include_kind: Optional[bool]=None) -> None:
    """
        Updates the component schema registry
        """
    from lazyops.libs.abcs.utils.helpers import flatten_dict_value
    include_kind = include_kind if include_kind is not None else self.include_kind_in_component_name
    if include_kind:
        prefix = f'{self.module_name}.{kind}' if kind else self.module_name
    else:
        prefix = self.module_name
    mapping = flatten_dict_value(components, prefix)
    self._component_schema_registry.update(mapping)