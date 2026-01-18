from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_register_empty_prefix_condition_with_colon(self):
    checker = checkers.Checker()
    checker.namespace().register('testns', '')
    with self.assertRaises(checkers.RegisterError) as ctx:
        checker.register('x:y', 'testns', lambda x: None)
    self.assertEqual(ctx.exception.args[0], 'caveat condition x:y in namespace testns contains a colon but its prefix is empty')