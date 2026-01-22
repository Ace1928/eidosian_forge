import socket
import time
from urllib.parse import urlparse
import dns.asyncbackend
import dns.inet
import dns.name
import dns.nameserver
import dns.query
import dns.rdtypes.svcbbase
def _extract_nameservers_from_svcb(answer):
    bootstrap_address = answer.nameserver
    if not dns.inet.is_address(bootstrap_address):
        return []
    infos = []
    for rr in answer.rrset.processing_order():
        nameservers = []
        param = rr.params.get(dns.rdtypes.svcbbase.ParamKey.ALPN)
        if param is None:
            continue
        alpns = set(param.ids)
        host = rr.target.to_text(omit_final_dot=True)
        port = None
        param = rr.params.get(dns.rdtypes.svcbbase.ParamKey.PORT)
        if param is not None:
            port = param.port
        if b'h2' in alpns:
            param = rr.params.get(dns.rdtypes.svcbbase.ParamKey.DOHPATH)
            if param is None or not param.value.endswith(b'{?dns}'):
                continue
            path = param.value[:-6].decode()
            if not path.startswith('/'):
                path = '/' + path
            if port is None:
                port = 443
            url = f'https://{host}:{port}{path}'
            try:
                urlparse(url)
                nameservers.append(dns.nameserver.DoHNameserver(url, bootstrap_address))
            except Exception:
                pass
        if b'dot' in alpns:
            if port is None:
                port = 853
            nameservers.append(dns.nameserver.DoTNameserver(bootstrap_address, port, host))
        if b'doq' in alpns:
            if port is None:
                port = 853
            nameservers.append(dns.nameserver.DoQNameserver(bootstrap_address, port, True, host))
        if len(nameservers) > 0:
            infos.append(_SVCBInfo(bootstrap_address, port, host, nameservers))
    return infos