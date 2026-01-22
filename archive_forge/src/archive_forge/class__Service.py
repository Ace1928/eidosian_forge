import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
class _Service(object):
    """Represents a service that requires authorization.

    Clients can make requests to the service to perform operations
    and may receive a macaroon to discharge if the authorization
    process requires it.
    """

    def __init__(self, name, auth, idm, locator):
        self._name = name
        self._store = _MacaroonStore(bakery.generate_key(), locator)
        self._checker = bakery.Checker(checker=test_checker(), authorizer=auth, identity_client=idm, macaroon_opstore=self._store)

    def name(self):
        return self._name

    def do(self, ctx, ms, ops):
        try:
            authInfo = self._checker.auth(ms).allow(ctx, ops)
        except bakery.DischargeRequiredError as exc:
            self._discharge_required_error(exc)
        return authInfo

    def do_any(self, ctx, ms, ops):
        try:
            authInfo, allowed = self._checker.auth(ms).allow_any(ctx, ops)
            return (authInfo, allowed)
        except bakery.DischargeRequiredError as exc:
            self._discharge_required_error(exc)

    def capability(self, ctx, ms, ops):
        try:
            conds = self._checker.auth(ms).allow_capability(ctx, ops)
        except bakery.DischargeRequiredError as exc:
            self._discharge_required_error(exc)
        m = self._store.new_macaroon(None, self._checker.namespace(), ops)
        for cond in conds:
            m.macaroon.add_first_party_caveat(cond)
        return m

    def _discharge_required_error(self, err):
        m = self._store.new_macaroon(err.cavs(), self._checker.namespace(), err.ops())
        name = 'authz'
        if len(err.ops()) == 1 and err.ops()[0] == bakery.LOGIN_OP:
            name = 'authn'
        raise _DischargeRequiredError(name=name, m=m)