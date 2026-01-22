from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress, IPNetwork
def _nmap_octet_target_values(spec):
    values = set()
    for element in spec.split(','):
        if '-' in element:
            left, right = element.split('-', 1)
            if not left:
                left = 0
            if not right:
                right = 255
            low = int(left)
            high = int(right)
            if not (0 <= low <= 255 and 0 <= high <= 255):
                raise ValueError('octet value overflow for spec %s!' % (spec,))
            if low > high:
                raise ValueError('left side of hyphen must be <= right %r' % (element,))
            for octet in range(low, high + 1):
                values.add(octet)
        else:
            octet = int(element)
            if not 0 <= octet <= 255:
                raise ValueError('octet value overflow for spec %s!' % (spec,))
            values.add(octet)
    return sorted(values)