from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from .. import util
from ..operations import ops
@_traverse.dispatch_for(ops.MigrationScript)
def _traverse_script(self, context: MigrationContext, revision: _GetRevArg, directive: MigrationScript) -> None:
    upgrade_ops_list: List[UpgradeOps] = []
    for upgrade_ops in directive.upgrade_ops_list:
        ret = self._traverse_for(context, revision, upgrade_ops)
        if len(ret) != 1:
            raise ValueError('Can only return single object for UpgradeOps traverse')
        upgrade_ops_list.append(ret[0])
    directive.upgrade_ops = upgrade_ops_list
    downgrade_ops_list: List[DowngradeOps] = []
    for downgrade_ops in directive.downgrade_ops_list:
        ret = self._traverse_for(context, revision, downgrade_ops)
        if len(ret) != 1:
            raise ValueError('Can only return single object for DowngradeOps traverse')
        downgrade_ops_list.append(ret[0])
    directive.downgrade_ops = downgrade_ops_list