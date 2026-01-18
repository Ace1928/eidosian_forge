import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_not_in_from_function_and_args(self):
    arg = []
    kw = dict()
    fn = context.get_context_from_function_and_args
    ctx1 = context.get_context_from_function_and_args(fn, arg, kw)
    self.assertIsNone(ctx1)