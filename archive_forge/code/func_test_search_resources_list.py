from unittest import mock
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
def test_search_resources_list(self):
    self.session._get.side_effect = exceptions.ResourceNotFound
    self.session._list.return_value = [self.FakeResource(foo='bar')]
    ret = self.cloud.search_resources('mock_session.fake', 'fake_name')
    self.session._get.assert_called_with(self.FakeResource, 'fake_name')
    self.session._list.assert_called_with(self.FakeResource, name='fake_name')
    self.assertEqual(1, len(ret))
    self.assertEqual(self.FakeResource(foo='bar').to_dict(), ret[0].to_dict())