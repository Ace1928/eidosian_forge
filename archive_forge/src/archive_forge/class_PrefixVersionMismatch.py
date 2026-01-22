from oslo_utils import excutils
from neutron_lib._i18n import _
class PrefixVersionMismatch(BadRequest):
    message = _('Cannot mix IPv4 and IPv6 prefixes in a subnet pool.')