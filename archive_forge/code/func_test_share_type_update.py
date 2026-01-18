import copy
from unittest import mock
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_type as mshare_type
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_share_type_update(self):
    share_type = self._init_share('stack_share_type_update')
    share_type.client().share_types.create.return_value = mock.MagicMock(id='type_id')
    fake_share_type = mock.MagicMock()
    share_type.client().share_types.get.return_value = fake_share_type
    scheduler.TaskRunner(share_type.create)()
    updated_props = copy.deepcopy(share_type.properties.data)
    updated_props[mshare_type.ManilaShareType.EXTRA_SPECS] = {'fake_key': 'fake_value'}
    after = rsrc_defn.ResourceDefinition(share_type.name, share_type.type(), updated_props)
    scheduler.TaskRunner(share_type.update, after)()
    fake_share_type.unset_keys.assert_called_once_with({'test': 'test'})
    fake_share_type.set_keys.assert_called_with(updated_props[mshare_type.ManilaShareType.EXTRA_SPECS])