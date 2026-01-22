import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionPushMpls(NXActionMplsBase):
    """
        Push MPLS action

        This action pushes a new MPLS header to the packet.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          push_mpls:ethertype
        ..

        +-------------------------------+
        | **push_mpls**\\:\\ *ethertype*  |
        +-------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        ethertype        Ether type(The value must be either 0x8847 or 0x8848)
        ================ ======================================================

        .. NOTE::
            This actions is supported by
            ``OFPActionPushMpls``
            in OpenFlow1.2 or later.

        Example::

            match = parser.OFPMatch(dl_type=0x0800)
            actions += [parser.NXActionPushMpls(ethertype=0x8847)]
        """
    _subtype = nicira_ext.NXAST_PUSH_MPLS