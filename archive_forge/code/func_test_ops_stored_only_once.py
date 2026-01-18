import copy
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
def test_ops_stored_only_once(self):
    st = bakery.MemoryOpsStore()
    test_oven = bakery.Oven(ops_store=st)
    ops = [bakery.Op('one', 'read'), bakery.Op('one', 'write'), bakery.Op('two', 'read')]
    m = test_oven.macaroon(bakery.LATEST_VERSION, AGES, None, ops)
    got_ops, conds = test_oven.macaroon_ops([m.macaroon])
    self.assertEqual(bakery.canonical_ops(got_ops), bakery.canonical_ops(ops))
    ops = [bakery.Op('one', 'write'), bakery.Op('one', 'read'), bakery.Op('one', 'read'), bakery.Op('two', 'read')]
    test_oven.macaroon(bakery.LATEST_VERSION, AGES, None, ops)
    self.assertEqual(len(st._store), 1)