from unittest import mock
from openstack import exceptions
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import receiver as sr
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_recv_delete_not_found(self):
    self.senlin_mock.delete_receiver.side_effect = [exceptions.ResourceNotFound(http_status=404)]
    recv = self._create_recv(self.t)
    scheduler.TaskRunner(recv.delete)()
    self.senlin_mock.delete_receiver.assert_called_once_with(recv.resource_id)