import json
from openstack.object_store.v1 import container
from openstack.tests.unit import base
def _test_no_headers(self, sot, sot_call, sess_method):
    headers = {}
    self.register_uris([dict(method=sess_method, uri=self.container_endpoint, validate=dict(headers=headers))])
    sot_call(self.cloud.object_store)