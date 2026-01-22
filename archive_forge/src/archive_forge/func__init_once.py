from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def _init_once(self, ctx):
    self._auth_indexes = {}
    self._conditions = [None] * len(self._macaroons)
    for i, ms in enumerate(self._macaroons):
        try:
            ops, conditions = self.parent._macaroon_opstore.macaroon_ops(ms)
        except VerificationError as e:
            self._init_errors.append(str(e))
            continue
        except Exception as exc:
            raise AuthInitError(str(exc))
        self._conditions[i] = conditions
        is_login = False
        for op in ops:
            if op == LOGIN_OP:
                is_login = True
            else:
                if op not in self._auth_indexes:
                    self._auth_indexes[op] = []
                self._auth_indexes[op].append(i)
        if not is_login:
            continue
        declared, err = self._check_conditions(ctx, LOGIN_OP, conditions)
        if err is not None:
            self._init_errors.append('cannot authorize login macaroon: ' + err)
            continue
        if self._identity is not None:
            continue
        try:
            identity = self.parent._identity_client.declared_identity(ctx, declared)
        except IdentityError as exc:
            self._init_errors.append('cannot decode declared identity: {}'.format(exc.args[0]))
            continue
        if LOGIN_OP not in self._auth_indexes:
            self._auth_indexes[LOGIN_OP] = []
        self._auth_indexes[LOGIN_OP].append(i)
        self._identity = identity
    if self._identity is None:
        try:
            identity, cavs = self.parent._identity_client.identity_from_context(ctx)
        except IdentityError:
            self._init_errors.append('could not determine identity')
        if cavs is None:
            cavs = []
        self._identity, self._identity_caveats = (identity, cavs)
    return None