from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def _register_client(self, kind: Optional[str]=None, include_kind: Optional[bool]=None, **kwargs):
    """
        Registers the client
        """
    kind = kind or self.kind
    self.settings.ctx.register_client(self, kind=self.kind, include_kind=include_kind, **kwargs)