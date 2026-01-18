import json
from openstack.object_store.v1 import container
from openstack.tests.unit import base
def test_set_temp_url_key(self):
    sot = container.Container.new(name=self.container)
    key = self.getUniqueString()
    self.register_uris([dict(method='POST', uri=self.container_endpoint, status_code=204, validate=dict(headers={'x-container-meta-temp-url-key': key})), dict(method='HEAD', uri=self.container_endpoint, headers={'x-container-meta-temp-url-key': key})])
    sot.set_temp_url_key(self.cloud.object_store, key)
    self.assert_calls()