import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_std_exception(self):
    name = reflection.get_class_name(RuntimeError)
    self.assertEqual('RuntimeError', name)