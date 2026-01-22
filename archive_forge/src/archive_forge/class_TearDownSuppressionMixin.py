import warnings
from twisted.trial import unittest, util
class TearDownSuppressionMixin:

    def tearDown(self):
        self._emit()