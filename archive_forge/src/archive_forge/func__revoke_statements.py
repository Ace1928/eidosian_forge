from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
def _revoke_statements(self, grantee: Role, privileges: set[Privilege]) -> Sequence[TextClause]:
    """
        Generates a revoke statement to commit via SQL.
        :param grantee: The :class:`lazyops.libs.dbinit.entities.Role` to revoke privileges from.
        :param privileges: The set of :class:`lazyops.libs.dbinit.data_structures.Privilege` to revoke.
        :return: A Sequence of :class:`sqlalchemy.TextClause` that represent the desired revoke statements.
        """
    return [text(f'REVOKE {self._format_privileges(privileges)} ON {self.__class__.__name__} {self.encoded_grant_name} FROM {grantee.encoded_name}')]