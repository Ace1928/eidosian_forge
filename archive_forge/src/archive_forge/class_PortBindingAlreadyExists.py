from oslo_utils import excutils
from neutron_lib._i18n import _
class PortBindingAlreadyExists(Conflict):
    message = _('Binding for port %(port_id)s on host %(host)s already exists.')