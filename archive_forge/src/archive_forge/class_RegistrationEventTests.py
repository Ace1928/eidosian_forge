import unittest
class RegistrationEventTests(unittest.TestCase, _ConformsToIRegistrationEvent):

    def _getTargetClass(self):
        from zope.interface.interfaces import RegistrationEvent
        return RegistrationEvent

    def test___repr__(self):
        target = object()
        event = self._makeOne(target)
        r = repr(event)
        self.assertEqual(r.splitlines(), ['RegistrationEvent event:', repr(target)])