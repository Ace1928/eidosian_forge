from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class DoQNameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=853, verify: Union[bool, str]=True, server_hostname: Optional[str]=None):
        super().__init__(address, port)
        self.verify = verify
        self.server_hostname = server_hostname

    def kind(self):
        return 'DoQ'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return dns.query.quic(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, server_hostname=self.server_hostname)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.quic(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, server_hostname=self.server_hostname)