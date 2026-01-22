from __future__ import annotations
from typing import Sequence, Type
from lazyops.libs.dbinit.base import Inspector, TextClause, inspect, text, DeclarativeBase
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.database import Database
from lazyops.libs.dbinit.entities.database_entity import DatabaseEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.entities.schema import Schema
from lazyops.libs.dbinit.mixins.grantable import Grantable
from lazyops.libs.dbinit.mixins.sql import SQLBase

        The SQL statement that checks to see what grants exist.
        :return: A single :class:`sqlalchemy.TextClause` containing the SQL to check what grants exist on this entity.
        