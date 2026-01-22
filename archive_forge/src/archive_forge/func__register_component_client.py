from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def _register_component_client(self, *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None):
    """
        Registers a component client
        """
    return self.settings.ctx.register_component_client(self, *parts, kind=kind, include_kind=include_kind)