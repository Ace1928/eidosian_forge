import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_retry_detail_save_with_task_failure(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    rd = models.RetryDetail('retry-1', uuid=uuidutils.generate_uuid())
    fail = failure.Failure.from_exception(RuntimeError('fail'))
    rd.results.append((42, {'some-task': fail}))
    fd.add(rd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
        conn.update_flow_details(fd)
        conn.update_atom_details(rd)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
    fd2 = lb2.find(fd.uuid)
    rd2 = fd2.find(rd.uuid)
    self.assertIsInstance(rd2, models.RetryDetail)
    fail2 = rd2.results[0][1].get('some-task')
    self.assertIsInstance(fail2, failure.Failure)
    self.assertTrue(fail.matches(fail2))