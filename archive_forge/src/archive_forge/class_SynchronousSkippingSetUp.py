from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousSkippingSetUp(SkippingSetUpMixin, SynchronousTestCase):
    pass