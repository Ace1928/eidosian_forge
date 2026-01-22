import warnings
from twisted.trial import unittest, util
class SetUpSuppressionMixin:

    def setUp(self):
        self._emit()