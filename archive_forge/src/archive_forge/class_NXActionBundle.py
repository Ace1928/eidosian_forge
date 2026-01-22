import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionBundle(_NXActionBundleBase):
    """
        Select bundle link action

        This action selects bundle link based on the specified parameters.
        Please refer to the ovs-ofctl command manual for details.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          bundle(fields, basis, algorithm, slave_type, slaves:[ s1, s2,...])
        ..

        +-----------------------------------------------------------+
        | **bundle(**\\ *fields*\\, \\ *basis*\\, \\ *algorithm*\\,       |
        | *slave_type*\\, \\ *slaves*\\:[ \\ *s1*\\, \\ *s2*\\,...]\\ **)** |
        +-----------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        algorithm        One of NX_MP_ALG_*.
        fields           One of NX_HASH_FIELDS_*
        basis            Universal hash parameter
        slave_type       Type of slaves(must be NXM_OF_IN_PORT)
        n_slaves         Number of slaves
        ofs_nbits        Start and End for the OXM/NXM field. (must be zero)
        dst              OXM/NXM header for source field(must be zero)
        slaves           List of slaves
        ================ ======================================================


        Example::

            actions += [parser.NXActionBundle(
                            algorithm=nicira_ext.NX_MP_ALG_HRW,
                            fields=nicira_ext.NX_HASH_FIELDS_ETH_SRC,
                            basis=0,
                            slave_type=nicira_ext.NXM_OF_IN_PORT,
                            n_slaves=2,
                            ofs_nbits=0,
                            dst=0,
                            slaves=[2, 3])]
        """
    _subtype = nicira_ext.NXAST_BUNDLE

    def __init__(self, algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst, slaves):
        super(NXActionBundle, self).__init__(algorithm, fields, basis, slave_type, n_slaves, ofs_nbits=0, dst=0, slaves=slaves)