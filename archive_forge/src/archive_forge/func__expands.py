from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
@_ec_dispatcher.dispatch_for(ops.AddConstraintOp)
@_ec_dispatcher.dispatch_for(ops.CreateIndexOp)
@_ec_dispatcher.dispatch_for(ops.CreateTableOp)
@_ec_dispatcher.dispatch_for(ops.AddColumnOp)
def _expands(context, directive, phase):
    if phase == 'expand':
        return directive
    else:
        return None