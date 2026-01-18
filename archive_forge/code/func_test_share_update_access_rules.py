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
def test_share_update_access_rules(self):
    share = self._create_share('stack_share_update_access_rules')
    updated_share_props = copy.deepcopy(share.properties.data)
    updated_share_props[mshare.ManilaShare.ACCESS_RULES] = [{mshare.ManilaShare.ACCESS_TO: '127.0.0.2', mshare.ManilaShare.ACCESS_TYPE: 'ip', mshare.ManilaShare.ACCESS_LEVEL: 'ro'}]
    share.client().shares.deny.return_value = None
    current_rule = {mshare.ManilaShare.ACCESS_TO: '127.0.0.1', mshare.ManilaShare.ACCESS_TYPE: 'ip', mshare.ManilaShare.ACCESS_LEVEL: 'ro', 'id': 'test_access_rule'}
    rule_tuple = collections.namedtuple('DummyRule', list(current_rule.keys()))
    share.client().shares.access_list.return_value = [rule_tuple(**current_rule)]
    after = rsrc_defn.ResourceDefinition(share.name, share.type(), updated_share_props)
    scheduler.TaskRunner(share.update, after)()
    share.client().shares.access_list.assert_called_once_with(share.resource_id)
    share.client().shares.allow.assert_called_with(share=share.resource_id, access_type='ip', access='127.0.0.2', access_level='ro')
    share.client().shares.deny.assert_called_once_with(share=share.resource_id, id='test_access_rule')