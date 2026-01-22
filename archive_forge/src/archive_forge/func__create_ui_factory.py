import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def _create_ui_factory(self):
    self.overrideEnv('TERM', None)
    return TextUIFactory('', StringIOAsTTY(), StringIOAsTTY())