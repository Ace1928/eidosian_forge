import copy
from unittest import mock
from ironicclient.common.apiclient import exceptions as ic_exc
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import ironic as ic
from heat.engine import resource
from heat.engine.resources.openstack.ironic import port
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _port_update(self, exc_msg=None):
    b = self._create_resource('port', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(b.create)()
    if exc_msg:
        self.client.port.update.side_effect = ic_exc.Conflict(exc_msg)
    t = template_format.parse(port_template)
    new_t = copy.deepcopy(t)
    new_extra = {'foo': 'bar'}
    m_pg = mock.Mock(extra=new_extra)
    self.client.port.get.return_value = m_pg
    new_t['resources'][self.fake_name]['properties']['extra'] = new_extra
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_port = rsrc_defns[self.fake_name]
    if exc_msg:
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(b.update, new_port))
        self.assertIn(exc_msg, str(exc))
    else:
        scheduler.TaskRunner(b.update, new_port)()
        self.client.port.update.assert_called_once_with(self.resource_id, [{'op': 'replace', 'path': '/extra', 'value': new_extra}])