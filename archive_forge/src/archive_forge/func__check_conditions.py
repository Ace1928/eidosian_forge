from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def _check_conditions(self, ctx, op, conds):
    declared = checkers.infer_declared_from_conditions(conds, self.parent.namespace())
    ctx = checkers.context_with_operations(ctx, [op.action])
    ctx = checkers.context_with_declared(ctx, declared)
    for cond in conds:
        err = self.parent._first_party_caveat_checker.check_first_party_caveat(ctx, cond)
        if err is not None:
            return (None, err)
    return (declared, None)