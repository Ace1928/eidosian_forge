from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError

        Wrapper around a dictionary to map from a letter code to a typed privilege.
        :param code: A letter code representing a privilege. See `Postgres docs <https://www.postgresql.org/docs/current/ddl-priv.html#PRIVILEGE-ABBREVS-TABLE>`_ for more.
        :return: A :class:`lazyops.libs.dbinit.data_structures.Privilege` corresponding to the provided letter code.
        