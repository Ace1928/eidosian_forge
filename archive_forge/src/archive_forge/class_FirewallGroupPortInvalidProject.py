from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupPortInvalidProject(exceptions.Conflict):
    message = _('Operation cannot be performed as port %(port_id)s is in an invalid project %(project_id)s.')