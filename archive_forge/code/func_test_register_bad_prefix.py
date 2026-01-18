from unittest import TestCase
import macaroonbakery.checkers as checkers
def test_register_bad_prefix(self):
    ns = checkers.Namespace(None)
    with self.assertRaises(ValueError):
        ns.register('std', 'x:1')