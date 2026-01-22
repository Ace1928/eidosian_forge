import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXFlowSpecLoad(_NXFlowSpec):
    """
        Add NXAST_REG_LOAD actions

        This class is used by ``NXActionLearn``.

        For the usage of this class, please refer to ``NXActionLearn``.

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        src              OXM/NXM header and Start bit for source field
        dst              OXM/NXM header and Start bit for destination field
        n_bits           The number of bits from the start bit
        ================ ======================================================
        """
    _dst_type = 1