import warnings
from twisted.trial import unittest, util
class SynchronousTestTearDownSuppression(TearDownSuppressionMixin, SynchronousTestSuppression):
    pass