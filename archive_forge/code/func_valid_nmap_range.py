from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress, IPNetwork
def valid_nmap_range(target_spec):
    """
    :param target_spec: an nmap-style IP range target specification.

    :return: ``True`` if IP range target spec is valid, ``False`` otherwise.
    """
    try:
        next(_parse_nmap_target_spec(target_spec))
        return True
    except (TypeError, ValueError, AddrFormatError):
        pass
    return False