import dns.rdtypes.dnskeybase
from dns.rdtypes.dnskeybase import flags_to_text_set, flags_from_text_set
class CDNSKEY(dns.rdtypes.dnskeybase.DNSKEYBase):
    """CDNSKEY record"""