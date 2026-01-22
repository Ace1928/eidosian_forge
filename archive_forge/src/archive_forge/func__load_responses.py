import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def _load_responses(self, responses):
    self.factory.responses.extend(responses)