import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_required_only(self):
    result = reflection.get_callable_args(function_with_defs, required_only=True)
    self.assertEqual(['a', 'b'], result)