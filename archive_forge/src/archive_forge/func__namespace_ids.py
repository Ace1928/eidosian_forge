from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
def _namespace_ids(self):
    return [id(n) for n in self.namespaces]