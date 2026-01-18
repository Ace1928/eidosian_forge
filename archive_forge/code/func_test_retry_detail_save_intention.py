import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_retry_detail_save_intention(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    rd = models.RetryDetail('retry-1', uuid=uuidutils.generate_uuid())
    fd.add(rd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
        conn.update_flow_details(fd)
        conn.update_atom_details(rd)
    rd.intention = states.REVERT
    with contextlib.closing(self._get_connection()) as conn:
        conn.update_atom_details(rd)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
    fd2 = lb2.find(fd.uuid)
    rd2 = fd2.find(rd.uuid)
    self.assertEqual(states.REVERT, rd2.intention)
    self.assertIsInstance(rd2, models.RetryDetail)