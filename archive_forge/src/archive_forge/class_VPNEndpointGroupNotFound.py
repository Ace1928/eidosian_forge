from neutron_lib._i18n import _
from neutron_lib import exceptions
class VPNEndpointGroupNotFound(exceptions.NotFound):
    message = _('Endpoint group %(endpoint_group_id)s could not be found')