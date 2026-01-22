import functools
from oslotest import base as test_base
from oslo_utils import reflection
class BadClass(object):

    def do_something(self):
        pass

    def __nonzero__(self):
        return False