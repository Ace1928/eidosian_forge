from neutron_lib._i18n import _
from neutron_lib import exceptions
class MixedIPVersionsForPeerCidrs(exceptions.BadRequest):
    message = _('Peer CIDRs do not have the same IP version, as required for IPSec site-to-site connection')