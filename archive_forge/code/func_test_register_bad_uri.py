from unittest import TestCase
import macaroonbakery.checkers as checkers
def test_register_bad_uri(self):
    ns = checkers.Namespace(None)
    with self.assertRaises(KeyError):
        ns.register('', 'x')