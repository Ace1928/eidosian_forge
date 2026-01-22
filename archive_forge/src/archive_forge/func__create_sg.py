import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_sg(self, name):
    if name:
        sg = sg_template['resources']['ServerGroup']
        sg['properties']['name'] = name
        self._init_template(sg_template)
        self.sg_mgr.create.return_value = FakeGroup(name)
    else:
        try:
            sg = sg_template['resources']['ServerGroup']
            del sg['properties']['name']
        except Exception:
            pass
        self._init_template(sg_template)
        name = 'test'
        n = name

        def fake_create(name, policy, rules):
            self.assertGreater(len(name), 1)
            return FakeGroup(n)
        self.sg_mgr.create = fake_create
    scheduler.TaskRunner(self.sg.create)()
    self.assertEqual((self.sg.CREATE, self.sg.COMPLETE), self.sg.state)
    self.assertEqual(name, self.sg.resource_id)