import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_qualified_class(self):

    class QualifiedClass(object):
        pass
    name = reflection.get_class_name(QualifiedClass)
    self.assertEqual('.'.join((__name__, 'QualifiedClass')), name)