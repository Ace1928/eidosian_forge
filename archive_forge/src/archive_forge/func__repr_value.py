import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
@property
def _repr_value(self) -> Optional[str]:
    """Safely returns the human-readable resource name if present."""
    if self._repr_attr is None or not hasattr(self, self._repr_attr):
        return None
    repr_val = getattr(self, self._repr_attr)
    if TYPE_CHECKING:
        assert isinstance(repr_val, str)
    return repr_val