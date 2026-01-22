from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class ReduceFunctorTaskTest(test.TestCase):

    def test_invalid_functor(self):
        self.assertRaises(ValueError, task.ReduceFunctorTask, 2, requires=5)
        self.assertRaises(ValueError, task.ReduceFunctorTask, lambda: None, requires=5)
        self.assertRaises(ValueError, task.ReduceFunctorTask, lambda x: None, requires=5)

    def test_functor_invalid_requires(self):
        self.assertRaises(TypeError, task.ReduceFunctorTask, lambda x, y: None, requires=1)
        self.assertRaises(ValueError, task.ReduceFunctorTask, lambda x, y: None, requires=[1])