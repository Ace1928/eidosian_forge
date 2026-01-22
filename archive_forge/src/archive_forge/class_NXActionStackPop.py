import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionStackPop(NXActionStackBase):
    """
        Pop field action

        This action pops field from top of the stack.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          pop:src[start...end]
        ..

        +----------------------------------------------------+
        | **pop**\\:\\ *src*\\ **[**\\ *start*\\...\\ *end*\\ **]** |
        +----------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        field            OXM/NXM header for destination field
        start            Start bit for destination field
        end              End bit for destination field
        ================ ======================================================

        Example::

            actions += [parser.NXActionStackPop(field="reg2",
                                                start=0,
                                                end=5)]
        """
    _subtype = nicira_ext.NXAST_STACK_POP