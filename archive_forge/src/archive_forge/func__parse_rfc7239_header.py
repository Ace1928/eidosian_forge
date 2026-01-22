from oslo_config import cfg
from oslo_middleware import base
@staticmethod
def _parse_rfc7239_header(header):
    """Parses RFC7239 Forward headers.

        e.g. for=192.0.2.60;proto=http, for=192.0.2.60;by=203.0.113.43

        """
    result = []
    for proxy in header.split(','):
        entry = {}
        for d in proxy.split(';'):
            key, _, value = d.partition('=')
            entry[key.lower().strip()] = value.strip()
        result.append(entry)
    return result