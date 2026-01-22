import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
class TestSilentUIFactory(tests.TestCase, UIFactoryTestMixin):

    def setUp(self):
        super().setUp()
        self.factory = ui.SilentUIFactory()

    def _check_note(self, note_text):
        pass

    def _check_show_error(self, msg):
        pass

    def _check_show_message(self, msg):
        pass

    def _check_show_warning(self, msg):
        pass

    def _check_log_transport_activity_noarg(self):
        pass

    def _check_log_transport_activity_display(self):
        pass

    def _check_log_transport_activity_display_no_bytes(self):
        pass

    def _load_responses(self, responses):
        pass