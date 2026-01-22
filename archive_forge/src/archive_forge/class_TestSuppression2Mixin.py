import warnings
from twisted.trial import unittest, util
class TestSuppression2Mixin(EmitMixin):

    def testSuppressModule(self):
        self._emit()