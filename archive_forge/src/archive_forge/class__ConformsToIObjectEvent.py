import unittest
class _ConformsToIObjectEvent:

    def _makeOne(self, target=None):
        if target is None:
            target = object()
        return self._getTargetClass()(target)

    def test_class_conforms_to_IObjectEvent(self):
        from zope.interface.interfaces import IObjectEvent
        from zope.interface.verify import verifyClass
        verifyClass(IObjectEvent, self._getTargetClass())

    def test_instance_conforms_to_IObjectEvent(self):
        from zope.interface.interfaces import IObjectEvent
        from zope.interface.verify import verifyObject
        verifyObject(IObjectEvent, self._makeOne())