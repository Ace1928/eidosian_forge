import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_transport_activity(self):
    t = transport.get_transport_from_url('memory:///')
    self.factory.report_transport_activity(t, 1000, 'write')
    self.factory.report_transport_activity(t, 2000, 'read')
    self.factory.report_transport_activity(t, 4000, None)
    self.factory.log_transport_activity()
    self._check_log_transport_activity_noarg()
    self.factory.log_transport_activity(display=True)
    self._check_log_transport_activity_display()