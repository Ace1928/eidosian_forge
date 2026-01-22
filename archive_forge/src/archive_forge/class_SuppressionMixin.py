import warnings
from twisted.trial import unittest, util
class SuppressionMixin(EmitMixin):
    suppress = [util.suppress(message=CLASS_WARNING_MSG)]

    def testSuppressMethod(self):
        self._emit()
    testSuppressMethod.suppress = [util.suppress(message=METHOD_WARNING_MSG)]

    def testSuppressClass(self):
        self._emit()

    def testOverrideSuppressClass(self):
        self._emit()
    testOverrideSuppressClass.suppress = []