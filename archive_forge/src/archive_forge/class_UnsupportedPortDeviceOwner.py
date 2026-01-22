from oslo_utils import excutils
from neutron_lib._i18n import _
class UnsupportedPortDeviceOwner(Conflict):
    message = _('Operation %(op)s is not supported for device_owner %(device_owner)s on port %(port_id)s.')