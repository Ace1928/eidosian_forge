import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetMplsLabel(NXAction):
    """
        Set MPLS Lavel action

        This action sets the MPLS Label.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          set_mpls_label:label
        ..

        +-------------------------------+
        | **set_mpls_label**\\:\\ *label* |
        +-------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        label            MPLS Label
        ================ ======================================================

        .. NOTE::
            This actions is supported by
            ``OFPActionSetField(mpls_label=label)``
            in OpenFlow1.2 or later.

        Example::

            actions += [parser.NXActionSetMplsLabel(label=0x10)]
        """
    _subtype = nicira_ext.NXAST_SET_MPLS_LABEL
    _fmt_str = '!2xI'

    def __init__(self, label, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionSetMplsLabel, self).__init__()
        self.label = label

    @classmethod
    def parser(cls, buf):
        label, = struct.unpack_from(cls._fmt_str, buf)
        return cls(label)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.label)
        return data