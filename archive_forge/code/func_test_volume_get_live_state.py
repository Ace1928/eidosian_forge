import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_volume_get_live_state(self):
    tmpl = "\n        heat_template_version: 2013-05-23\n        description: Cinder volume\n        resources:\n          volume:\n            type: OS::Cinder::Volume\n            properties:\n              size: 1\n              name: test_name\n              description: test_description\n              image: 1234\n              scheduler_hints:\n                'consistencygroup_id': 4444\n        "
    t = template_format.parse(tmpl)
    stack = utils.parse_stack(t, stack_name='get_live_state')
    rsrc = stack['volume']
    rsrc._availability_zone = 'nova'
    rsrc.resource_id = '1234'
    vol_resp = {'attachments': [], 'availability_zone': 'nova', 'snapshot_id': None, 'size': 1, 'metadata': {'test': 'test_value', 'readonly': False}, 'consistencygroup_id': '4444', 'volume_image_metadata': {'image_id': '1234', 'image_name': 'test'}, 'description': None, 'source_volid': None, 'name': 'test-volume-jbdbgdsy3vyg', 'volume_type': 'lvmdriver-1'}
    vol = mock.MagicMock()
    vol.to_dict.return_value = vol_resp
    rsrc.client().volumes = mock.MagicMock()
    rsrc.client().volumes.get = mock.MagicMock(return_value=vol)
    rsrc.client().volume_api_version = 3
    rsrc.data = mock.MagicMock(return_value={'volume_type': 'lvmdriver-1'})
    reality = rsrc.get_live_state(rsrc.properties)
    expected = {'size': 1, 'metadata': {'test': 'test_value'}, 'description': None, 'name': 'test-volume-jbdbgdsy3vyg', 'backup_id': None, 'read_only': False}
    self.assertEqual(expected, reality)