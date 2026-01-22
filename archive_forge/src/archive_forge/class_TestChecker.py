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
class TestChecker(TestCase):

    def setUp(self):
        self._discharges = []

    def test_authorize_with_open_access_and_no_macaroons(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {bakery.EVERYONE}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        auth_info = client.do(test_context, ts, [bakery.Op(entity='something', action='read')])
        self.assertEqual(len(self._discharges), 0)
        self.assertIsNotNone(auth_info)
        self.assertIsNone(auth_info.identity)
        self.assertEqual(len(auth_info.macaroons), 0)

    def test_authorization_denied(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = bakery.ClosedAuthorizer()
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        with self.assertRaises(bakery.PermissionDenied):
            client.do(ctx, ts, [bakery.Op(entity='something', action='read')])

    def test_authorize_with_authentication_required(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        auth_info = client.do(ctx, ts, [bakery.Op(entity='something', action='read')])
        self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob')])
        self.assertIsNotNone(auth_info)
        self.assertEqual(auth_info.identity.id(), 'bob')
        self.assertEqual(len(auth_info.macaroons), 1)

    def test_authorize_multiple_ops(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {'bob'}, bakery.Op(entity='otherthing', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        client.do(ctx, ts, [bakery.Op(entity='something', action='read'), bakery.Op(entity='otherthing', action='read')])
        self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob')])

    def test_capability(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m = client.discharged_capability(ctx, ts, [bakery.Op(entity='something', action='read')])
        auth_info = ts.do(test_context, [m], [bakery.Op(entity='something', action='read')])
        self.assertIsNotNone(auth_info)
        self.assertIsNone(auth_info.identity)
        self.assertEqual(len(auth_info.macaroons), 1)
        self.assertEqual(auth_info.macaroons[0][0].identifier_bytes, m[0].identifier_bytes)

    def test_capability_multiple_entities(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'bob'}, bakery.Op(entity='e2', action='read'): {'bob'}, bakery.Op(entity='e3', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m = client.discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read'), bakery.Op(entity='e3', action='read')])
        self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob')])
        ts.do(test_context, [m], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read'), bakery.Op(entity='e3', action='read')])
        ts.do(test_context, [m], [bakery.Op(entity='e2', action='read'), bakery.Op(entity='e3', action='read')])
        ts.do(test_context, [m], [bakery.Op(entity='e3', action='read')])

    def test_multiple_capabilities(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m1 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m2 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e2', action='read')])
        self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='alice'), _DischargeRecord(location='ids', user='bob')])
        auth_info = ts.do(test_context, [m1, m2], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
        self.assertIsNotNone(auth_info)
        self.assertIsNone(auth_info.identity)
        self.assertEqual(len(auth_info.macaroons), 2)
        self.assertEqual(auth_info.macaroons[0][0].identifier_bytes, m1[0].identifier_bytes)
        self.assertEqual(auth_info.macaroons[1][0].identifier_bytes, m2[0].identifier_bytes)

    def test_combine_capabilities(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}, bakery.Op(entity='e3', action='read'): {'bob', 'alice'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m1 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e3', action='read')])
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m2 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e2', action='read')])
        m = ts.capability(test_context, [m1, m2], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read'), bakery.Op(entity='e3', action='read')])
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read'), bakery.Op(entity='e3', action='read')])

    def test_partially_authorized_request(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        client = _Client(locator)
        client.add_macaroon(ts, 'authz', m)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        client.discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])

    def test_auth_with_third_party_caveats(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)

        def authorize_with_tp_discharge(ctx, id, op):
            if id is not None and id.id() == 'bob' and (op == bakery.Op(entity='something', action='read')):
                return (True, [checkers.Caveat(condition='question', location='other third party')])
            return (False, None)
        auth = bakery.AuthorizerFunc(authorize_with_tp_discharge)
        ts = _Service('myservice', auth, ids, locator)

        class _LocalDischargeChecker(bakery.ThirdPartyCaveatChecker):

            def check_third_party_caveat(_, ctx, info):
                if info.condition != 'question':
                    raise ValueError('third party condition not recognized')
                self._discharges.append(_DischargeRecord(location='other third party', user=ctx.get(_DISCHARGE_USER_KEY)))
                return []
        locator['other third party'] = _Discharger(key=bakery.generate_key(), checker=_LocalDischargeChecker(), locator=locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        client.do(ctx, ts, [bakery.Op(entity='something', action='read')])
        self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob'), _DischargeRecord(location='other third party', user='bob')])

    def test_capability_combines_first_party_caveats(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m1 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        m1.macaroon.add_first_party_caveat('true 1')
        m1.macaroon.add_first_party_caveat('true 2')
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m2 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e2', action='read')])
        m2.macaroon.add_first_party_caveat('true 3')
        m2.macaroon.add_first_party_caveat('true 4')
        client = _Client(locator)
        client.add_macaroon(ts, 'authz1', [m1.macaroon])
        client.add_macaroon(ts, 'authz2', [m2.macaroon])
        m = client.capability(test_context, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
        self.assertEqual(_macaroon_conditions(m.macaroon.caveats, False), ['true 1', 'true 2', 'true 3', 'true 4'])

    def test_first_party_caveat_squashing(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'alice'}})
        ts = _Service('myservice', auth, ids, locator)
        tests = [('duplicates removed', [checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 2', namespace='testns'), checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 3', namespace='testns')], [checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 2', namespace='testns'), checkers.Caveat(condition='true 3', namespace='testns')]), ('earliest time before', [checkers.time_before_caveat(epoch + timedelta(days=1)), checkers.Caveat(condition='true 1', namespace='testns'), checkers.time_before_caveat(epoch + timedelta(days=0, hours=1)), checkers.time_before_caveat(epoch + timedelta(days=0, hours=0, minutes=5))], [checkers.time_before_caveat(epoch + timedelta(days=0, hours=0, minutes=5)), checkers.Caveat(condition='true 1', namespace='testns')]), ('operations and declared caveats removed', [checkers.deny_caveat(['foo']), checkers.allow_caveat(['read', 'write']), checkers.declared_caveat('username', 'bob'), checkers.Caveat(condition='true 1', namespace='testns')], [checkers.Caveat(condition='true 1', namespace='testns')])]
        for test in tests:
            print(test[0])
            ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
            m1 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
            m1.add_caveats(test[1], None, None)
            m2 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
            m2.add_caveat(checkers.Caveat(condition='true notused', namespace='testns'), None, None)
            client = _Client(locator)
            client.add_macaroon(ts, 'authz1', [m1.macaroon])
            client.add_macaroon(ts, 'authz2', [m2.macaroon])
            m3 = client.capability(test_context, ts, [bakery.Op(entity='e1', action='read')])
            self.assertEqual(_macaroon_conditions(m3.macaroon.caveats, False), _resolve_caveats(m3.namespace, test[2]))

    def test_login_only(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = bakery.ClosedAuthorizer()
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        auth_info = _Client(locator).do(ctx, ts, [bakery.LOGIN_OP])
        self.assertIsNotNone(auth_info)
        self.assertEqual(auth_info.identity.id(), 'bob')

    def test_allow_any(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        client = _Client(locator)
        client.add_macaroon(ts, 'authz', m)
        self._discharges = []
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        with self.assertRaises(_DischargeRequiredError):
            client.do_any(ctx, ts, [bakery.LOGIN_OP, bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='read')])
            self.assertEqual(len(self._discharges), 0)
        _, err = client.do(ctx, ts, [bakery.LOGIN_OP])
        auth_info, allowed = client.do_any(ctx, ts, [bakery.LOGIN_OP, bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='read')])
        self.assertEqual(auth_info.identity.id(), 'bob')
        self.assertEqual(len(auth_info.macaroons), 2)
        self.assertEqual(allowed, [True, True, True])

    def test_auth_with_identity_from_context(self):
        locator = _DischargerLocator()
        ids = _BasicAuthIdService()
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'sherlock'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        ctx = _context_with_basic_auth(test_context, 'sherlock', 'holmes')
        auth_info = _Client(locator).do(ctx, ts, [bakery.Op(entity='e1', action='read')])
        self.assertEqual(auth_info.identity.id(), 'sherlock')
        self.assertEqual(len(auth_info.macaroons), 0)

    def test_auth_login_op_with_identity_from_context(self):
        locator = _DischargerLocator()
        ids = _BasicAuthIdService()
        ts = _Service('myservice', bakery.ClosedAuthorizer(), ids, locator)
        ctx = _context_with_basic_auth(test_context, 'sherlock', 'holmes')
        auth_info = _Client(locator).do(ctx, ts, [bakery.LOGIN_OP])
        self.assertEqual(auth_info.identity.id(), 'sherlock')
        self.assertEqual(len(auth_info.macaroons), 0)

    def test_operation_allow_caveat(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'bob'}, bakery.Op(entity='e1', action='write'): {'bob'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m = client.capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='write'), bakery.Op(entity='e2', action='read')])
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])
        m.add_caveat(checkers.allow_caveat(['read']), None, None)
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
        with self.assertRaises(_DischargeRequiredError):
            ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])

    def test_operation_deny_caveat(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'bob'}, bakery.Op(entity='e1', action='write'): {'bob'}, bakery.Op(entity='e2', action='read'): {'bob'}})
        ts = _Service('myservice', auth, ids, locator)
        client = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        m = client.capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='write'), bakery.Op(entity='e2', action='read')])
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])
        m.add_caveat(checkers.deny_caveat(['write']), None, None)
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
        with self.assertRaises(_DischargeRequiredError):
            ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])

    def test_duplicate_login_macaroons(self):
        locator = _DischargerLocator()
        ids = _IdService('ids', locator, self)
        auth = bakery.ClosedAuthorizer()
        ts = _Service('myservice', auth, ids, locator)
        client1 = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
        auth_info = client1.do(ctx, ts, [bakery.LOGIN_OP])
        self.assertEqual(auth_info.identity.id(), 'bob')
        client2 = _Client(locator)
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        auth_info = client2.do(ctx, ts, [bakery.LOGIN_OP])
        self.assertEqual(auth_info.identity.id(), 'alice')
        client3 = _Client(locator)
        client3.add_macaroon(ts, '1.bob', client1._macaroons[ts.name()]['authn'])
        client3.add_macaroon(ts, '2.alice', client2._macaroons[ts.name()]['authn'])
        auth_info = client3.do(test_context, ts, [bakery.LOGIN_OP])
        self.assertEqual(auth_info.identity.id(), 'bob')
        self.assertEqual(len(auth_info.macaroons), 1)
        client3 = _Client(locator)
        client3.add_macaroon(ts, '1.alice', client2._macaroons[ts.name()]['authn'])
        client3.add_macaroon(ts, '2.bob', client1._macaroons[ts.name()]['authn'])
        auth_info = client3.do(test_context, ts, [bakery.LOGIN_OP])
        self.assertEqual(auth_info.identity.id(), 'alice')
        self.assertEqual(len(auth_info.macaroons), 1)

    def test_macaroon_ops_fatal_error(self):
        checker = bakery.Checker(macaroon_opstore=_MacaroonStoreWithError())
        m = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2)
        with self.assertRaises(bakery.AuthInitError):
            checker.auth([m]).allow(test_context, [bakery.LOGIN_OP])