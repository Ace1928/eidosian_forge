from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.entities.database import Database
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.mixins.sql import SQLCreatable

        :param name: Unique name of the entity. Cluster-level entities must be unique, database-level entities must be unique within a database.
        :param database: The :class:`lazyops.libs.dbinit.entities.Database` that this entity belongs to.
        :param depends_on: Any entities that should be created before this one.
        :param check_if_exists: Flag to set existence check behavior. If `True`, will raise an exception during _safe_create if the entity already exists, and will raise an exception during _safe_drop if the entity does not exist.
        