from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.base import Engine, TextClause, create_engine, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.mixins.grantable import Grantable
def _create_statements(self) -> Sequence[TextClause]:
    statement = f'CREATE DATABASE {self.encoded_name}'
    props = self._get_passed_args()
    for k, v in props.items():
        if k == 'owner' and isinstance(v, Role):
            statement = f'{statement} OWNER={v.encoded_name}'
        elif k != 'grants':
            statement = f'{statement} {k.upper()}={v}'
    return [text(statement)]