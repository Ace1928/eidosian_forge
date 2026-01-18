from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_register_twice_same_namespace(self):
    checker = checkers.Checker()
    checker.namespace().register('testns', '')
    checker.register('x', 'testns', lambda x: None)
    with self.assertRaises(checkers.RegisterError) as ctx:
        checker.register('x', 'testns', lambda x: None)
    self.assertEqual(ctx.exception.args[0], 'checker for x (namespace testns) already registered in namespace testns')