import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import domain_configs
def _assert_resource_attributes(self, resource, req_ref):
    for attr in req_ref:
        self.assertEqual(getattr(resource, attr), req_ref[attr], 'Expected different %s' % attr)