import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
class TestProviderAPIRegistry(unit.BaseTestCase):

    def setUp(self):
        super(TestProviderAPIRegistry, self).setUp()
        provider_api.ProviderAPIs._clear_registry_instances()
        self.addCleanup(provider_api.ProviderAPIs._clear_registry_instances)

    def _create_manager_instance(self, provides_api=None):
        provides_api = provides_api or '%s_api' % uuid.uuid4().hex

        class TestManager(manager.Manager):
            _provides_api = provides_api
            driver_namespace = '_TEST_NOTHING'

            def do_something(self):
                return provides_api
        return TestManager(driver_name=None)

    def test_deferred_gettr(self):
        api_name = '%s_api' % uuid.uuid4().hex

        class TestClass(object):
            descriptor = provider_api.ProviderAPIs.deferred_provider_lookup(api=api_name, method='do_something')
        test_instance = TestClass()
        self.assertRaises(AttributeError, getattr, test_instance, 'descriptor')
        self._create_manager_instance(provides_api=api_name)
        self.assertEqual(api_name, test_instance.descriptor())

    def test_registry_lock(self):
        provider_api.ProviderAPIs.lock_provider_registry()
        self.assertRaises(RuntimeError, self._create_manager_instance)

    def test_registry_duplicate(self):
        test_manager = self._create_manager_instance()
        self.assertRaises(provider_api.DuplicateProviderError, self._create_manager_instance, provides_api=test_manager._provides_api)

    def test_provider_api_mixin(self):
        test_manager = self._create_manager_instance()

        class Testing(provider_api.ProviderAPIMixin, object):
            pass
        instance = Testing()
        self.assertIs(test_manager, getattr(instance, test_manager._provides_api))

    def test_manager_api_reference(self):
        manager = self._create_manager_instance()
        second_manager = self._create_manager_instance()
        self.assertIs(second_manager, getattr(manager, second_manager._provides_api))
        self.assertIs(manager, getattr(second_manager, manager._provides_api))