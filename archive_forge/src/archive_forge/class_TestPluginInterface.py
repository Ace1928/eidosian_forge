import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
class TestPluginInterface(test_base.BaseTestCase):

    class ServicePluginStub(base.ServicePluginBase):

        def get_plugin_type(self):
            pass

        def get_plugin_description(self):
            pass

    def test_issubclass_hook(self):

        class A(TestPluginInterface.ServicePluginStub):

            def f(self):
                pass

        class B(base.ServicePluginBase):

            @abc.abstractmethod
            def f(self):
                pass
        self.assertTrue(issubclass(A, B))

    def test_issubclass_hook_class_without_abstract_methods(self):

        class A(object):

            def f(self):
                pass

        class B(base.ServicePluginBase):

            def f(self):
                pass
        self.assertFalse(issubclass(A, B))

    def test_issubclass_hook_not_all_methods_implemented(self):

        class A(TestPluginInterface.ServicePluginStub):

            def f(self):
                pass

        class B(base.ServicePluginBase):

            @abc.abstractmethod
            def f(self):
                pass

            @abc.abstractmethod
            def g(self):
                pass
        self.assertFalse(issubclass(A, B))