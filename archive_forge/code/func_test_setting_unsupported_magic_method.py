from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_setting_unsupported_magic_method(self):
    mock = MagicMock()

    def set_setattr():
        mock.__setattr__ = lambda self, name: None
    self.assertRaisesRegex(AttributeError, "Attempting to set unsupported magic method '__setattr__'.", set_setattr)