from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.base import Engine, TextClause, create_engine, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.mixins.grantable import Grantable
def _grants_exist(self, grantee: Role, privileges: set[Privilege]) -> bool:
    rows = self._fetch_sql(engine=self.engine(), statement=self._grants_exist_statement())
    try:
        existing_privileges = list(filter(None, [self._extract_privileges(acl=r[0], grantee=grantee) for r in rows]))[0]
        return privileges.issubset(existing_privileges)
    except IndexError:
        return False