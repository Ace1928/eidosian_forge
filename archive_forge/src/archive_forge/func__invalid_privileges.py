from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
def _invalid_privileges(self, privileges: set[Privilege]) -> set[Privilege]:
    """
        Find all invalid privileges for this entity type.
        :param privileges: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` to check for invalid entries.
        :return: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` that are invalid. Empty if all valid.
        """
    return privileges.difference(self._allowed_privileges())