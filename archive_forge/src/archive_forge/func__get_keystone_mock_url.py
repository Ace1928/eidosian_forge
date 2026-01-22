import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def _get_keystone_mock_url(self, resource, append=None, v3=True, qs_elements=None):
    base_url_append = None
    if v3:
        base_url_append = 'v3'
    return self.get_mock_url(service_type='identity', resource=resource, append=append, base_url_append=base_url_append, qs_elements=qs_elements)