from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
class NeutronExtConstraint(NeutronConstraint):

    def validate_with_client(self, client, value):
        neutron_plugin = client.client_plugin(CLIENT_NAME)
        if self.extension and (not neutron_plugin.has_extension(self.extension)):
            raise exception.EntityNotFound(entity='neutron extension', name=self.extension)
        neutron_plugin.resolve_ext_resource(self.resource_name, value)