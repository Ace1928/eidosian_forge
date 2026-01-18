from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.data_structures.privileges import Privilege

    Represents a Sequence of :class:`lazyops.libs.dbinit.data_structures.Privilege` to grant to a
    :class:`lazyops.libs.dbinit.entities.Role` for a given :class:`lazyops.libs.dbinit.mixins.Grantable`.
    