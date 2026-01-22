from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidSharedSetting(Conflict):
    message = _('Unable to reconfigure sharing settings for network %(network)s. Multiple tenants are using it.')