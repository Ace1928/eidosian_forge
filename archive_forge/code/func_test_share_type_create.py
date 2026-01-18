import copy
from unittest import mock
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_type as mshare_type
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_share_type_create(self):
    share_type = self._init_share('stack_share_type_create')
    fake_share_type = mock.MagicMock(id='type_id')
    share_type.client().share_types.create.return_value = fake_share_type
    scheduler.TaskRunner(share_type.create)()
    self.assertEqual('type_id', share_type.resource_id)
    share_type.client().share_types.create.assert_called_once_with(name='test_share_type', spec_driver_handles_share_servers=True, is_public=False, spec_snapshot_support=True)
    fake_share_type.set_keys.assert_called_once_with({'test': 'test'})
    self.assertEqual('share_types', share_type.entity)