import dns.immutable
import dns.rdtypes.txtbase
@dns.immutable.immutable
class NINFO(dns.rdtypes.txtbase.TXTBase):
    """NINFO record"""