import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_share_update_metadata(self):
    share = self._create_share('stack_share_update_metadata')
    updated_share_props = copy.deepcopy(share.properties.data)
    updated_share_props[mshare.ManilaShare.METADATA] = {'fake_key': 'fake_value'}
    share.client().shares.update_all_metadata.return_value = None
    after = rsrc_defn.ResourceDefinition(share.name, share.type(), updated_share_props)
    scheduler.TaskRunner(share.update, after)()
    share.client().shares.update_all_metadata.assert_called_once_with(share.resource_id, updated_share_props[mshare.ManilaShare.METADATA])