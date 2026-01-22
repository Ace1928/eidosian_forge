import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
class FlowSpecL2VPNNLRI(_FlowSpecNLRIBase):
    """
    Flow Specification NLRI class for L2VPN [draft-ietf-idr-flowspec-l2vpn-05]
    """
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    FLOWSPEC_FAMILY = 'l2vpnfs'

    def __init__(self, length=0, route_dist=None, rules=None):
        super(FlowSpecL2VPNNLRI, self).__init__(length, rules)
        assert route_dist is not None
        self.route_dist = route_dist

    @classmethod
    def _from_user(cls, route_dist, **kwargs):
        rules = []
        for k, v in kwargs.items():
            subcls = _FlowSpecComponentBase.lookup_type_name(k, cls.ROUTE_FAMILY.afi)
            rule = subcls.from_str(str(v))
            rules.extend(rule)
        rules.sort(key=lambda x: x.type)
        return cls(route_dist=route_dist, rules=rules)

    @classmethod
    def from_user(cls, route_dist, **kwargs):
        """
        Utility method for creating a L2VPN NLRI instance.

        This function returns a L2VPN NLRI instance
        from human readable format value.

        :param kwargs: The following arguments are available.

        ============== ============= ========= ==============================
        Argument       Value         Operator  Description
        ============== ============= ========= ==============================
        ether_type     Integer       Numeric   Ethernet Type.
        src_mac        Mac Address   Nothing   Source Mac address.
        dst_mac        Mac Address   Nothing   Destination Mac address.
        llc_ssap       Integer       Numeric   Source Service Access Point
                                               in LLC.
        llc_dsap       Integer       Numeric   Destination Service Access
                                               Point in LLC.
        llc_control    Integer       Numeric   Control field in LLC.
        snap           Integer       Numeric   Sub-Network Access Protocol
                                               field.
        vlan_id        Integer       Numeric   VLAN ID.
        vlan_cos       Integer       Numeric   VLAN COS field.
        inner_vlan_id  Integer       Numeric   Inner VLAN ID.
        inner_vlan_cos Integer       Numeric   Inner VLAN COS field.
        ============== ============= ========= ==============================
        """
        return cls._from_user(route_dist, **kwargs)

    @property
    def formatted_nlri_str(self):
        return '%s:%s' % (self.route_dist, self.prefix)