from typing import Any, Dict, List, Optional, Generic, TypeVar, cast
from types import TracebackType
from importlib.metadata import entry_points
from toolz import curry
Return the currently active plugin.