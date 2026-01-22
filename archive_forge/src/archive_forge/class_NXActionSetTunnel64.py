import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetTunnel64(_NXActionSetTunnelBase):
    """
        Set Tunnel action

        This action outputs to a port that encapsulates
        the packet in a tunnel.

        And equivalent to the followings action of ovs-ofctl command.

        .. note::
            This actions is supported by
            ``OFPActionSetField``
            in OpenFlow1.2 or later.

        ..
          set_tunnel64:id
        ..

        +--------------------------+
        | **set_tunnel64**\\:\\ *id* |
        +--------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        tun_id           Tunnel ID(64bits)
        ================ ======================================================

        Example::

            actions += [parser.NXActionSetTunnel64(tun_id=0xa)]
        """
    _subtype = nicira_ext.NXAST_SET_TUNNEL64
    _fmt_str = '!6xQ'