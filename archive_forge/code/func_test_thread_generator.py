import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
def test_thread_generator(self):
    model = os_tgen.ThreadReportGenerator()()
    self.assertTrue(len(model.keys()) >= 1)
    was_ok = False
    for val in model.values():
        self.assertIsInstance(val, os_tmod.ThreadModel)
        self.assertIsNotNone(val.stack_trace)
        if val.thread_id == threading.current_thread().ident:
            was_ok = True
            break
    self.assertTrue(was_ok)
    model.set_current_view_type('text')
    self.assertIsNotNone(str(model))