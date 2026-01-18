import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
def test_thread_generator_tb(self):

    class FakeModel(object):

        def __init__(self, thread_id, tb):
            self.traceback = tb
    with mock.patch('oslo_reports.models.threading.ThreadModel', FakeModel):
        model = os_tgen.ThreadReportGenerator('fake traceback')()
        curr_thread = model.get(threading.current_thread().ident, None)
        self.assertIsNotNone(curr_thread, None)
        self.assertEqual('fake traceback', curr_thread.traceback)