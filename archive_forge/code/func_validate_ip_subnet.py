import netaddr
from neutronclient._i18n import _
from neutronclient.common import exceptions
def validate_ip_subnet(parsed_args, attr_name):
    val = getattr(parsed_args, attr_name)
    if not val:
        return
    try:
        netaddr.IPNetwork(val)
    except (netaddr.AddrFormatError, ValueError):
        raise exceptions.CommandError(_('%(attr_name)s "%(val)s" is not a valid CIDR.') % {'attr_name': attr_name.replace('_', '-'), 'val': val})