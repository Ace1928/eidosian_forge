from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
class _APIDefinition(extensions.APIExtensionDescriptor):
    api_definition = TestAPIExtensionDescriptor