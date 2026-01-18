from unittest import mock
from urllib import parse as urlparse
from heat.common import template_format
from heat.engine.clients import client_plugin
from heat.engine import resource
from heat.engine.resources.openstack.zaqar import queue
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_update_in_place(self):
    t = template_format.parse(wp_template)
    self.parse_stack(t)
    queue = self.stack['MyQueue2']
    queue.resource_id_set(queue.properties.get('name'))
    fake_q = FakeQueue('myqueue', auto_create=False)
    self.fc.queue.return_value = fake_q
    t = template_format.parse(wp_template)
    new_queue = t['Resources']['MyQueue2']
    new_queue['Properties']['metadata'] = {'key1': 'value'}
    resource_defns = template.Template(t).resource_definitions(self.stack)
    scheduler.TaskRunner(queue.create)()
    self.fc.queue.assert_called_once_with('myqueue', auto_create=False)
    fake_q.metadata.assert_called_with(new_meta={'key1': {'key2': 'value', 'key3': [1, 2]}})
    scheduler.TaskRunner(queue.update, resource_defns['MyQueue2'])()
    fake_q.metadata.assert_called_with(new_meta={'key1': 'value'})