import functools
from oslotest import base as test_base
from oslo_utils import reflection
class GetClassNameTest(test_base.BaseTestCase):

    def test_std_exception(self):
        name = reflection.get_class_name(RuntimeError)
        self.assertEqual('RuntimeError', name)

    def test_class(self):
        name = reflection.get_class_name(Class)
        self.assertEqual('.'.join((__name__, 'Class')), name)

    def test_qualified_class(self):

        class QualifiedClass(object):
            pass
        name = reflection.get_class_name(QualifiedClass)
        self.assertEqual('.'.join((__name__, 'QualifiedClass')), name)

    def test_instance(self):
        name = reflection.get_class_name(Class())
        self.assertEqual('.'.join((__name__, 'Class')), name)

    def test_int(self):
        name = reflection.get_class_name(42)
        self.assertEqual('int', name)

    def test_class_method(self):
        name = reflection.get_class_name(Class.class_method)
        self.assertEqual('%s.Class' % __name__, name)
        name = reflection.get_class_name(Class.class_method, fully_qualified=False)
        self.assertEqual('Class', name)

    def test_static_method(self):
        self.assertRaises(TypeError, reflection.get_class_name, Class.static_method)

    def test_unbound_method(self):
        self.assertRaises(TypeError, reflection.get_class_name, mere_function)

    def test_bound_method(self):
        c = Class()
        name = reflection.get_class_name(c.method)
        self.assertEqual('%s.Class' % __name__, name)
        name = reflection.get_class_name(c.method, fully_qualified=False)
        self.assertEqual('Class', name)