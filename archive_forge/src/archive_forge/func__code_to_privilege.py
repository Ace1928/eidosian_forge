from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
@staticmethod
def _code_to_privilege(code: str) -> Privilege:
    """
        Wrapper around a dictionary to map from a letter code to a typed privilege.
        :param code: A letter code representing a privilege. See `Postgres docs <https://www.postgresql.org/docs/current/ddl-priv.html#PRIVILEGE-ABBREVS-TABLE>`_ for more.
        :return: A :class:`lazyops.libs.dbinit.data_structures.Privilege` corresponding to the provided letter code.
        """
    return {'r': Privilege.SELECT, 'w': Privilege.UPDATE, 'a': Privilege.INSERT, 'd': Privilege.DELETE, 'D': Privilege.TRUNCATE, 'x': Privilege.REFERENCES, 't': Privilege.TRIGGER, 'X': Privilege.EXECUTE, 'U': Privilege.USAGE, 'C': Privilege.CREATE, 'c': Privilege.CONNECT, 'T': Privilege.TEMPORARY}[code]