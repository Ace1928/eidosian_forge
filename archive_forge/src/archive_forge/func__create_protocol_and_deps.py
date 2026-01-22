import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def _create_protocol_and_deps(self):
    identity_provider = unit.new_identity_provider_ref()
    identity_provider = PROVIDERS.federation_api.create_idp(identity_provider['id'], identity_provider)
    mapping = PROVIDERS.federation_api.create_mapping(uuid.uuid4().hex, unit.new_mapping_ref())
    protocol = unit.new_protocol_ref(mapping_id=mapping['id'])
    protocol = PROVIDERS.federation_api.create_protocol(identity_provider['id'], protocol['id'], protocol)
    return (protocol, mapping, identity_provider)