from taskflow import test
from taskflow.utils import async_utils as au
class MakeCompletedFutureTest(test.TestCase):

    def test_make_completed_future(self):
        result = object()
        future = au.make_completed_future(result)
        self.assertTrue(future.done())
        self.assertIs(future.result(), result)