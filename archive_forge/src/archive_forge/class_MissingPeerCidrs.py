from neutron_lib._i18n import _
from neutron_lib import exceptions
class MissingPeerCidrs(exceptions.BadRequest):
    message = _('Missing peer CIDRs for IPsec site-to-site connection')