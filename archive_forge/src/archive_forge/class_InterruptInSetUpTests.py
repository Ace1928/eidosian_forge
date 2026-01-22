from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
class InterruptInSetUpTests(TrialTest):
    testsRun = 0
    test_02_run: bool

    class InterruptedTest(unittest.TestCase):

        def setUp(self) -> None:
            if InterruptInSetUpTests.testsRun > 0:
                raise KeyboardInterrupt

        def test_01(self) -> None:
            InterruptInSetUpTests.testsRun += 1

        def test_02(self) -> None:
            InterruptInSetUpTests.testsRun += 1
            InterruptInSetUpTests.test_02_run = True

    def setUp(self) -> None:
        super().setUp()
        self.suite = self.loader.loadClass(InterruptInSetUpTests.InterruptedTest)
        InterruptInSetUpTests.test_02_run = False
        InterruptInSetUpTests.testsRun = 0

    def test_setUpOK(self) -> None:
        self.assertEqual(0, InterruptInSetUpTests.testsRun)
        self.assertEqual(2, self.suite.countTestCases())
        self.assertEqual(0, self.reporter.testsRun)
        self.assertFalse(self.reporter.shouldStop)

    def test_interruptInSetUp(self) -> None:
        runner.TrialSuite([self.suite]).run(self.reporter)
        self.assertTrue(self.reporter.shouldStop)
        self.assertEqual(2, self.reporter.testsRun)
        self.assertFalse(InterruptInSetUpTests.test_02_run, 'test_02 ran')