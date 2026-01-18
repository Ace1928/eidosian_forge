import unittest
from unittest.test.support import LoggingResult
def test_skip_doesnt_run_setup(self):

    class Foo(unittest.TestCase):
        wasSetUp = False
        wasTornDown = False

        def setUp(self):
            Foo.wasSetUp = True

        def tornDown(self):
            Foo.wasTornDown = True

        @unittest.skip('testing')
        def test_1(self):
            pass
    result = unittest.TestResult()
    test = Foo('test_1')
    suite = unittest.TestSuite([test])
    self.assertIs(suite.run(result), result)
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertFalse(Foo.wasSetUp)
    self.assertFalse(Foo.wasTornDown)