import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSample(NXAction):
    """
        Sample packets action

        This action samples packets and sends one sample for
        every sampled packet.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          sample(argument[,argument]...)
        ..

        +----------------------------------------------------+
        | **sample(**\\ *argument*\\[,\\ *argument*\\]...\\ **)** |
        +----------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        probability      The number of sampled packets
        collector_set_id The unsigned 32-bit integer identifier of
                         the set of sample collectors to send sampled packets
                         to
        obs_domain_id    The Unsigned 32-bit integer Observation Domain ID
        obs_point_id     The unsigned 32-bit integer Observation Point ID
        ================ ======================================================

        Example::

            actions += [parser.NXActionSample(probability=3,
                                              collector_set_id=1,
                                              obs_domain_id=2,
                                              obs_point_id=3,)]
        """
    _subtype = nicira_ext.NXAST_SAMPLE
    _fmt_str = '!HIII'

    def __init__(self, probability, collector_set_id=0, obs_domain_id=0, obs_point_id=0, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionSample, self).__init__()
        self.probability = probability
        self.collector_set_id = collector_set_id
        self.obs_domain_id = obs_domain_id
        self.obs_point_id = obs_point_id

    @classmethod
    def parser(cls, buf):
        probability, collector_set_id, obs_domain_id, obs_point_id = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(probability, collector_set_id, obs_domain_id, obs_point_id)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.probability, self.collector_set_id, self.obs_domain_id, self.obs_point_id)
        return data