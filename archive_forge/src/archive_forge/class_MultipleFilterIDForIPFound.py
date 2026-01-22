from oslo_utils import excutils
from neutron_lib._i18n import _
class MultipleFilterIDForIPFound(Conflict):
    message = _('Multiple filter IDs for IP %(ip)s found.')