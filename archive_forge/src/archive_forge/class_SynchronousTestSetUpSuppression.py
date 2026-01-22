import warnings
from twisted.trial import unittest, util
class SynchronousTestSetUpSuppression(SetUpSuppressionMixin, SynchronousTestSuppression):
    pass