from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
def with_outer_namespace(self, outer_namespace):
    """
        Return a new Environment with an extra namespace added.

        This namespace will be used only for variables that are not found
        in any existing namespace, i.e., it is "outside" them all.
        """
    return Environment(self.namespaces + [outer_namespace])