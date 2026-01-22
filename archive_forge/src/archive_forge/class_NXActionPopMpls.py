import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionPopMpls(NXActionMplsBase):
    """
        Pop MPLS action

        This action pops the MPLS header from the packet.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          pop_mpls:ethertype
        ..

        +------------------------------+
        | **pop_mpls**\\:\\ *ethertype*  |
        +------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        ethertype        Ether type
        ================ ======================================================

        .. NOTE::
            This actions is supported by
            ``OFPActionPopMpls``
            in OpenFlow1.2 or later.

        Example::

            match = parser.OFPMatch(dl_type=0x8847)
            actions += [parser.NXActionPushMpls(ethertype=0x0800)]
        """
    _subtype = nicira_ext.NXAST_POP_MPLS