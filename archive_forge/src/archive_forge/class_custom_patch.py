import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
class custom_patch(_patch):

    def __exit__(self, etype=None, val=None, tb=None):
        _patch.__exit__(self, etype, val, tb)
        holder.exc_info = (etype, val, tb)
    stop = __exit__