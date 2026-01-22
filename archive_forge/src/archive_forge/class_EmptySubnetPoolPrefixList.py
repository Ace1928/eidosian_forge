from oslo_utils import excutils
from neutron_lib._i18n import _
class EmptySubnetPoolPrefixList(BadRequest):
    message = _('Empty subnet pool prefix list.')