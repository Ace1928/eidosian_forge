from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def _get_component_schema(self, name: str, *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None) -> Type['ComponentSchemaT']:
    """
        Gets a component schema
        """
    return self.settings.ctx.get_component_schema(name, *parts, kind=kind, include_kind=include_kind)