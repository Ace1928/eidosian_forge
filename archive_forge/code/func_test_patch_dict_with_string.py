import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_dict_with_string(self):

    @patch.dict('os.environ', {'konrad_delong': 'some value'})
    def test():
        self.assertIn('konrad_delong', os.environ)
    test()