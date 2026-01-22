from oslo_utils import excutils
from neutron_lib._i18n import _
class PortBindingError(NeutronException):
    message = _('Binding for port %(port_id)s on host %(host)s could not be created or updated.')