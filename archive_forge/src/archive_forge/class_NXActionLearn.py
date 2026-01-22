import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionLearn(NXAction):
    """
        Adds or modifies flow action

        This action adds or modifies a flow in OpenFlow table.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          learn(argument[,argument]...)
        ..

        +---------------------------------------------------+
        | **learn(**\\ *argument*\\[,\\ *argument*\\]...\\ **)** |
        +---------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        table_id         The table in which the new flow should be inserted
        specs            Adds a match criterion to the new flow

                         Please use the
                         ``NXFlowSpecMatch``
                         in order to set the following format

                         ..
                           field=value
                           field[start..end]=src[start..end]
                           field[start..end]
                         ..

                         | *field*\\=\\ *value*
                         | *field*\\ **[**\\ *start*\\..\\ *end*\\ **]**\\  =\\ *src*\\ **[**\\ *start*\\..\\ *end*\\ **]**
                         | *field*\\ **[**\\ *start*\\..\\ *end*\\ **]**
                         |

                         Please use the
                         ``NXFlowSpecLoad``
                         in order to set the following format

                         ..
                           load:value->dst[start..end]
                           load:src[start..end]->dst[start..end]
                         ..

                         | **load**\\:\\ *value*\\ **->**\\ *dst*\\ **[**\\ *start*\\..\\ *end*\\ **]**
                         | **load**\\:\\ *src*\\ **[**\\ *start*\\..\\ *end*\\ **] ->**\\ *dst*\\ **[**\\ *start*\\..\\ *end*\\ **]**
                         |

                         Please use the
                         ``NXFlowSpecOutput``
                         in order to set the following format

                         ..
                           output:field[start..end]
                         ..

                         | **output:**\\ field\\ **[**\\ *start*\\..\\ *end*\\ **]**

        idle_timeout     Idle time before discarding(seconds)
        hard_timeout     Max time before discarding(seconds)
        priority         Priority level of flow entry
        cookie           Cookie for new flow
        flags            send_flow_rem
        fin_idle_timeout Idle timeout after FIN(seconds)
        fin_hard_timeout Hard timeout after FIN(seconds)
        ================ ======================================================

        .. CAUTION::
            The arguments specify the flow's match fields, actions,
            and other properties, as follows.
            At least one match criterion and one action argument
            should ordinarily be specified.

        Example::

            actions += [
                parser.NXActionLearn(able_id=10,
                     specs=[parser.NXFlowSpecMatch(src=0x800,
                                                   dst=('eth_type_nxm', 0),
                                                   n_bits=16),
                            parser.NXFlowSpecMatch(src=('reg1', 1),
                                                   dst=('reg2', 3),
                                                   n_bits=5),
                            parser.NXFlowSpecMatch(src=('reg3', 1),
                                                   dst=('reg3', 1),
                                                   n_bits=5),
                            parser.NXFlowSpecLoad(src=0,
                                                  dst=('reg4', 3),
                                                  n_bits=5),
                            parser.NXFlowSpecLoad(src=('reg5', 1),
                                                  dst=('reg6', 3),
                                                  n_bits=5),
                            parser.NXFlowSpecOutput(src=('reg7', 1),
                                                    dst="",
                                                    n_bits=5)],
                     idle_timeout=180,
                     hard_timeout=300,
                     priority=1,
                     cookie=0x64,
                     flags=ofproto.OFPFF_SEND_FLOW_REM,
                     fin_idle_timeout=180,
                     fin_hard_timeout=300)]
        """
    _subtype = nicira_ext.NXAST_LEARN
    _fmt_str = '!HHHQHBxHH'

    def __init__(self, table_id, specs, idle_timeout=0, hard_timeout=0, priority=ofp.OFP_DEFAULT_PRIORITY, cookie=0, flags=0, fin_idle_timeout=0, fin_hard_timeout=0, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionLearn, self).__init__()
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.priority = priority
        self.cookie = cookie
        self.flags = flags
        self.table_id = table_id
        self.fin_idle_timeout = fin_idle_timeout
        self.fin_hard_timeout = fin_hard_timeout
        self.specs = specs

    @classmethod
    def parser(cls, buf):
        idle_timeout, hard_timeout, priority, cookie, flags, table_id, fin_idle_timeout, fin_hard_timeout = struct.unpack_from(cls._fmt_str, buf, 0)
        rest = buf[struct.calcsize(cls._fmt_str):]
        specs = []
        while len(rest) > 0:
            spec, rest = _NXFlowSpec.parse(rest)
            if spec is None:
                continue
            specs.append(spec)
        return cls(idle_timeout=idle_timeout, hard_timeout=hard_timeout, priority=priority, cookie=cookie, flags=flags, table_id=table_id, fin_idle_timeout=fin_idle_timeout, fin_hard_timeout=fin_hard_timeout, specs=specs)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.idle_timeout, self.hard_timeout, self.priority, self.cookie, self.flags, self.table_id, self.fin_idle_timeout, self.fin_hard_timeout)
        for spec in self.specs:
            data += spec.serialize()
        return data