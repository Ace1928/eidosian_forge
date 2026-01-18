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
def walk_network_integration_targets(include_hidden: bool=False) -> c.Iterable[IntegrationTarget]:
    """Return an iterable of network integration targets."""
    for target in walk_integration_targets():
        if 'network/' in target.aliases or (include_hidden and 'hidden/network/' in target.aliases):
            yield target