from unittest import TestCase
import macaroonbakery.checkers as checkers
def test_register(self):
    ns = checkers.Namespace(None)
    ns.register('testns', 't')
    prefix = ns.resolve('testns')
    self.assertEqual(prefix, 't')
    ns.register('other', 'o')
    prefix = ns.resolve('other')
    self.assertEqual(prefix, 'o')
    ns.register('other', 'p')
    prefix = ns.resolve('other')
    self.assertEqual(prefix, 'o')