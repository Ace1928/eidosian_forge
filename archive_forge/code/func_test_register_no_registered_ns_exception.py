from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_register_no_registered_ns_exception(self):
    checker = checkers.Checker()
    with self.assertRaises(checkers.RegisterError) as ctx:
        checker.register('x', 'testns', lambda x: None)
    self.assertEqual(ctx.exception.args[0], 'no prefix registered for namespace testns when registering condition x')