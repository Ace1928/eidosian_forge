from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_acls_lazy(self):
    data = self.container.get_dict(self.entity_href, consumers=[self.container.consumer])
    m = self.responses.get(self.entity_href, json=data)
    acl_data = {'read': {'project-access': True, 'users': ['u2']}}
    acl_ref = self.entity_href + '/acl'
    n = self.responses.get(acl_ref, json=acl_data)
    container = self.manager.get(container_ref=self.entity_href)
    self.assertIsNotNone(container)
    self.assertEqual(self.container.name, container.name)
    self.assertTrue(m.called)
    self.assertFalse(n.called)
    self.assertEqual(['u2'], container.acls.read.users)
    self.assertTrue(container.acls.read.project_access)
    self.assertIsInstance(container.acls, acls.ContainerACL)
    self.assertEqual(acl_ref, n.last_request.url)