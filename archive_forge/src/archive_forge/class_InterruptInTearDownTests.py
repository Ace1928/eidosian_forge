from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
class InterruptInTearDownTests(TrialTest):
    testsRun = 0
    test_02_run: bool

    class InterruptedTest(unittest.TestCase):

        def tearDown(self) -> None:
            if InterruptInTearDownTests.testsRun > 0:
                raise KeyboardInterrupt

        def test_01(self) -> None:
            InterruptInTearDownTests.testsRun += 1

        def test_02(self) -> None:
            InterruptInTearDownTests.testsRun += 1
            InterruptInTearDownTests.test_02_run = True

    def setUp(self) -> None:
        super().setUp()
        self.suite = self.loader.loadClass(InterruptInTearDownTests.InterruptedTest)
        InterruptInTearDownTests.testsRun = 0
        InterruptInTearDownTests.test_02_run = False

    def test_setUpOK(self) -> None:
        self.assertEqual(0, InterruptInTearDownTests.testsRun)
        self.assertEqual(2, self.suite.countTestCases())
        self.assertEqual(0, self.reporter.testsRun)
        self.assertFalse(self.reporter.shouldStop)

    def test_interruptInTearDown(self) -> None:
        runner.TrialSuite([self.suite]).run(self.reporter)
        self.assertEqual(1, self.reporter.testsRun)
        self.assertTrue(self.reporter.shouldStop)
        self.assertFalse(InterruptInTearDownTests.test_02_run, 'test_02 ran')