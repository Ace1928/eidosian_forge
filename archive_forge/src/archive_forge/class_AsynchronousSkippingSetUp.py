from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class AsynchronousSkippingSetUp(SkippingSetUpMixin, TestCase):
    pass