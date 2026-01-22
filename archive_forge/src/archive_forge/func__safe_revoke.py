from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from typing import Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_on import GrantOn, GrantStore
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError
def _safe_revoke(self) -> None:
    """
        Performs an existence check before executing revoke statements in the cluster.
        """
    if self.grants:
        if not self._exists():
            raise EntityExistsError(f'There is no {self.__class__.__name__} with the name {self.name}. The {self.__class__.__name__} must exist to revoke privileges.')
        for target, privileges in self.grants.items():
            target._safe_revoke(grantee=self, privileges=privileges)