from os_ken.ofproto.oxx_fields import (
from os_ken.ofproto import ofproto_common
class NiciraExtended0(_OxmClass):
    """Nicira Extended Match (NXM_0)

    NXM header format is same as 32-bit (non-experimenter) OXMs.
    """
    _class = OFPXMC_NXM_0