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
class CompletionUnavailable(Completion):
    """Argument completion unavailable."""
    message: str = 'No completions available.'