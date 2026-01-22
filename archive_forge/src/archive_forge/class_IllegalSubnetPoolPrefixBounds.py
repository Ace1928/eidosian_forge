from oslo_utils import excutils
from neutron_lib._i18n import _
class IllegalSubnetPoolPrefixBounds(BadRequest):
    message = _('Illegal prefix bounds: %(prefix_type)s=%(prefixlen)s, %(base_prefix_type)s=%(base_prefixlen)s.')