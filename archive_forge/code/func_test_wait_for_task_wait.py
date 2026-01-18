import io
from unittest import mock
import requests
from openstack import exceptions
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task as _task
from openstack import proxy as proxy_base
from openstack.tests.unit.image.v2 import test_image as fake_image
from openstack.tests.unit import test_proxy_base
def test_wait_for_task_wait(self):
    res = _task.Task(id='id', status='waiting')
    mock_fetch = mock.Mock()
    mock_fetch.side_effect = [_task.Task(id='id', status='waiting'), _task.Task(id='id', status='waiting'), _task.Task(id='id', status='success')]
    with mock.patch.object(_task.Task, 'fetch', mock_fetch):
        result = self.proxy.wait_for_task(res, interval=0.01, wait=0.5)
        self.assertEqual('success', result.status)