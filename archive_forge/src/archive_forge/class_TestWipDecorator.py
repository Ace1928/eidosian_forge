from testtools import matchers
from testtools import testcase
from keystone.tests.unit import utils
class TestWipDecorator(testcase.TestCase):

    def test_raises_SkipError_when_broken_test_fails(self):

        @utils.wip('waiting on bug #000000')
        def test():
            raise Exception('i expected a failure - this is a WIP')
        e = self.assertRaises(testcase.TestSkipped, test)
        self.assertThat(str(e), matchers.Contains('#000000'))

    def test_raises_AssertionError_when_test_passes(self):

        @utils.wip('waiting on bug #000000')
        def test():
            pass
        e = self.assertRaises(AssertionError, test)
        self.assertThat(str(e), matchers.Contains('#000000'))