from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from heat.common import context
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import event as event_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(event_object.Event, 'get_all_by_tenant')
def test_tenant_events_list_with_marker_and_filters(self, mock_get_all):
    limit = object()
    marker = object()
    sort_keys = object()
    sort_dir = object()
    filters = {}
    self.eng.list_events(self.ctx, None, limit=limit, marker=marker, sort_keys=sort_keys, sort_dir=sort_dir, filters=filters)
    mock_get_all.assert_called_once_with(self.ctx, limit=limit, sort_keys=sort_keys, marker=marker, sort_dir=sort_dir, filters=filters)