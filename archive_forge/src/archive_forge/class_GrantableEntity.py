from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
class GrantableEntity(Grantable, Entity):
    """
    Convenience class that inherits from Grantable and Entity.
    """

    def grant(self, grants: Sequence[GrantTo]) -> None:
        super().grant(grants=grants)
        self._fix_entity_order(grants=grants, target_entity=self)