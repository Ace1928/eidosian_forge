from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class TearDownTodoMixin:

    def tearDown(self):
        raise RuntimeError('deliberate error')

    def test_todo1(self):
        pass
    test_todo1.todo = 'tearDown todo1'