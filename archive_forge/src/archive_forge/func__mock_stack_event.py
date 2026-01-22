from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@staticmethod
def _mock_stack_event(event_id, stack_name, stack_status='CREATE_COMPLETE'):
    stack_id = 'abcdef'
    ev_info = {'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/stacks/%s/%s' % (stack_name, stack_id), 'rel': 'stack'}], 'logical_resource_id': stack_name, 'physical_resource_id': stack_id, 'resource_name': stack_name, 'resource_status': stack_status, 'resource_status_reason': 'state changed', 'event_time': '2014-12-05T14:14:30Z', 'id': event_id}
    return hc_ev.Event(manager=None, info=ev_info)