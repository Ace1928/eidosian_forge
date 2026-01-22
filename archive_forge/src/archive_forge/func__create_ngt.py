from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_ngt(self, template):
    ngt = self._init_ngt(template)
    self.ngt_mgr.create.return_value = self.fake_ngt
    scheduler.TaskRunner(ngt.create)()
    self.assertEqual((ngt.CREATE, ngt.COMPLETE), ngt.state)
    self.assertEqual(self.fake_ngt.id, ngt.resource_id)
    return ngt