from neutron_lib._i18n import _
from neutron_lib import exceptions
class VPNServiceNotFound(exceptions.NotFound):
    message = _('VPNService %(vpnservice_id)s could not be found')