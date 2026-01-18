import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_std_class_up_to(self):
    names = list(reflection.get_all_class_names(RuntimeError, up_to=Exception))
    self.assertEqual(RUNTIME_ERROR_CLASSES[:-2], names)