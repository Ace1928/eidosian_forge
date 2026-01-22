import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionConjunction(NXAction):
    """
        Conjunctive matches action

        This action ties groups of individual OpenFlow flows into
        higher-level conjunctive flows.
        Please refer to the ovs-ofctl command manual for details.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          conjunction(id,k/n)
        ..

        +--------------------------------------------------+
        | **conjunction(**\\ *id*\\,\\ *k*\\ **/**\\ *n*\\ **)** |
        +--------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        clause           Number assigned to the flow's dimension
        n_clauses        Specify the conjunctive flow's match condition
        id\\_             Conjunction ID
        ================ ======================================================

        Example::

            actions += [parser.NXActionConjunction(clause=1,
                                                   n_clauses=2,
                                                   id_=10)]
        """
    _subtype = nicira_ext.NXAST_CONJUNCTION
    _fmt_str = '!BBI'

    def __init__(self, clause, n_clauses, id_, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionConjunction, self).__init__()
        self.clause = clause
        self.n_clauses = n_clauses
        self.id = id_

    @classmethod
    def parser(cls, buf):
        clause, n_clauses, id_ = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(clause, n_clauses, id_)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.clause, self.n_clauses, self.id)
        return data