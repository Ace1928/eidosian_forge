import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_with_skiplist(self):
    templ = copy.deepcopy(template_server)
    self.mock_flavor = mock.Mock(ram=4, disk=4)
    self.mock_active_image = mock.Mock(min_ram=1, min_disk=1, status='active')
    self.mock_inactive_image = mock.Mock(min_ram=1, min_disk=1, status='inactive')

    def get_image(image_identifier):
        if image_identifier == 'image0':
            return self.mock_inactive_image
        else:
            return self.mock_active_image
    self.patchobject(glance.GlanceClientPlugin, 'get_image', side_effect=get_image)
    self.patchobject(nova.NovaClientPlugin, 'get_flavor', return_value=self.mock_flavor)
    props = templ['resources']['group1']['properties']
    props['removal_policies'] = [{'resource_list': ['0']}]
    stack = utils.parse_stack(templ)
    resg = stack.resources['group1']
    self.assertIsNone(resg.validate())