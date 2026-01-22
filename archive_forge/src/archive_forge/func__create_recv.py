from unittest import mock
from openstack import exceptions
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import receiver as sr
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_recv(self, template):
    recv = self._init_recv(template)
    self.senlin_mock.create_receiver.return_value = self.fake_r
    self.senlin_mock.get_receiver.return_value = self.fake_r
    scheduler.TaskRunner(recv.create)()
    self.assertEqual((recv.CREATE, recv.COMPLETE), recv.state)
    self.assertEqual(self.fake_r.id, recv.resource_id)
    return recv