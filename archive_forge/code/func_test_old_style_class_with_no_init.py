import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.skipIf(six.PY3, 'no old style classes in Python 3')
def test_old_style_class_with_no_init(self):

    class Foo:
        pass
    create_autospec(Foo)