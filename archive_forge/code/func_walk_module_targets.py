from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def walk_module_targets() -> c.Iterable[TestTarget]:
    """Iterate through the module test targets."""
    for target in walk_test_targets(path=data_context().content.module_path, module_path=data_context().content.module_path, extensions=MODULE_EXTENSIONS):
        if not target.module:
            continue
        yield target