import dns.immutable
import dns.rdtypes.tlsabase
@dns.immutable.immutable
class SMIMEA(dns.rdtypes.tlsabase.TLSABase):
    """SMIMEA record"""