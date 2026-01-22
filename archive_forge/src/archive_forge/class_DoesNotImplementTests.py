import unittest
class DoesNotImplementTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.exceptions import DoesNotImplement
        return DoesNotImplement

    def _makeOne(self, *args):
        iface = _makeIface()
        return self._getTargetClass()(iface, *args)

    def test___str__(self):
        dni = self._makeOne()
        self.assertEqual(str(dni), 'An object has failed to implement interface zope.interface.tests.test_exceptions.IDummy: Does not declaratively implement the interface.')

    def test___str__w_candidate(self):
        dni = self._makeOne('candidate')
        self.assertEqual(str(dni), "The object 'candidate' has failed to implement interface zope.interface.tests.test_exceptions.IDummy: Does not declaratively implement the interface.")