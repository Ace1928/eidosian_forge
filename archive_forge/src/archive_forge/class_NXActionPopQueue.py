import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionPopQueue(NXAction):
    """
        Pop queue action

        This action restors the queue to the value it was before any
        set_queue actions were applied.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          pop_queue
        ..

        +---------------+
        | **pop_queue** |
        +---------------+

        Example::

            actions += [parser.NXActionPopQueue()]
        """
    _subtype = nicira_ext.NXAST_POP_QUEUE
    _fmt_str = '!6x'

    def __init__(self, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionPopQueue, self).__init__()

    @classmethod
    def parser(cls, buf):
        return cls()

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0)
        return data