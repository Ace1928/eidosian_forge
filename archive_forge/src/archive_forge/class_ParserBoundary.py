from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@dataclasses.dataclass
class ParserBoundary:
    """Boundary details for parsing composite input."""
    delimiters: str
    required: bool
    match: t.Optional[str] = None
    ready: bool = True