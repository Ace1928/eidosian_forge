import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
class LoadFromFactoryTestCase(test.TestCase):

    def test_non_reimportable(self):

        def factory():
            pass
        self.assertRaisesRegex(ValueError, 'Flow factory .* is not reimportable', taskflow.engines.load_from_factory, factory)

    def test_it_works(self):
        engine = taskflow.engines.load_from_factory(my_flow_factory, factory_kwargs={'task_name': 'test1'})
        self.assertIsInstance(engine._flow, test_utils.DummyTask)
        fd = engine.storage._flowdetail
        self.assertEqual('test1', fd.name)
        self.assertEqual({'name': '%s.my_flow_factory' % __name__, 'args': [], 'kwargs': {'task_name': 'test1'}}, fd.meta.get('factory'))

    def test_it_works_by_name(self):
        factory_name = '%s.my_flow_factory' % __name__
        engine = taskflow.engines.load_from_factory(factory_name, factory_kwargs={'task_name': 'test1'})
        self.assertIsInstance(engine._flow, test_utils.DummyTask)
        fd = engine.storage._flowdetail
        self.assertEqual('test1', fd.name)
        self.assertEqual({'name': factory_name, 'args': [], 'kwargs': {'task_name': 'test1'}}, fd.meta.get('factory'))