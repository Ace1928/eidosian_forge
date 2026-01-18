from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
Return the active rule names.