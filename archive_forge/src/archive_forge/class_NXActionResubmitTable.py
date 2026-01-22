import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionResubmitTable(NXAction):
    """
        Resubmit action

        This action searches one of the switch's flow tables.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          resubmit([port],[table])
        ..

        +------------------------------------------------+
        | **resubmit(**\\[\\ *port*\\]\\,[\\ *table*\\]\\ **)** |
        +------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        in_port          New in_port for checking flow table
        table_id         Checking flow tables
        ================ ======================================================

        Example::

            actions += [parser.NXActionResubmitTable(in_port=8080,
                                                     table_id=10)]
        """
    _subtype = nicira_ext.NXAST_RESUBMIT_TABLE
    _fmt_str = '!HB3x'

    def __init__(self, in_port=65528, table_id=255, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionResubmitTable, self).__init__()
        self.in_port = in_port
        self.table_id = table_id

    @classmethod
    def parser(cls, buf):
        in_port, table_id = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(in_port, table_id)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.in_port, self.table_id)
        return data