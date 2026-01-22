from os_ken.ofproto.oxx_fields import (
from os_ken.ofproto import ofproto_common
class NiciraExtended1(_OxmClass):
    """Nicira Extended Match (NXM_1)

    NXM header format is same as 32-bit (non-experimenter) OXMs.
    """
    _class = OFPXMC_NXM_1