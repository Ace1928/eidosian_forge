import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_task_detail_update_not_existing(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    td = models.TaskDetail('detail-1', uuid=uuidutils.generate_uuid())
    fd.add(td)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    td2 = models.TaskDetail('detail-1', uuid=uuidutils.generate_uuid())
    fd.add(td2)
    with contextlib.closing(self._get_connection()) as conn:
        conn.update_flow_details(fd)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb.uuid)
    fd2 = lb2.find(fd.uuid)
    self.assertIsNotNone(fd2.find(td.uuid))
    self.assertIsNotNone(fd2.find(td2.uuid))