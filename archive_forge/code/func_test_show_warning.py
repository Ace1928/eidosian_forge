import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_show_warning(self):
    msg = 'a warning'
    self.factory.show_warning(msg)
    self._check_show_warning(msg)