import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_note(self):
    self.factory.note('a note to the user')
    self._check_note('a note to the user')