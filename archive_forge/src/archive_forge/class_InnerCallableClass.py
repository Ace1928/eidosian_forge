import functools
from oslotest import base as test_base
from oslo_utils import reflection
class InnerCallableClass(object):

    def __call__(self):
        pass