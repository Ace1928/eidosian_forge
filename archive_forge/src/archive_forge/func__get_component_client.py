from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def _get_component_client(self, name: str, *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None) -> 'ClientTypes':
    """
        Gets a component client
        """
    return self.settings.ctx.get_component_client(name, *parts, kind=kind, include_kind=include_kind)