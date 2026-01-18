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
def walk_internal_targets(targets: c.Iterable[TCompletionTarget], includes: t.Optional[list[str]]=None, excludes: t.Optional[list[str]]=None, requires: t.Optional[list[str]]=None) -> tuple[TCompletionTarget, ...]:
    """Return a tuple of matching completion targets."""
    targets = tuple(targets)
    include_targets = sorted(filter_targets(targets, includes), key=lambda include_target: include_target.name)
    if requires:
        require_targets = set(filter_targets(targets, requires))
        include_targets = [require_target for require_target in include_targets if require_target in require_targets]
    if excludes:
        list(filter_targets(targets, excludes, include=False))
    internal_targets = set(filter_targets(include_targets, excludes, errors=False, include=False))
    return tuple(sorted(internal_targets, key=lambda sort_target: sort_target.name))