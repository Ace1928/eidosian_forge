import abc
import logging
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
class VpnPath(Path, metaclass=abc.ABCMeta):
    ROUTE_FAMILY = None
    VRF_PATH_CLASS = None
    NLRI_CLASS = None

    def clone_to_vrf(self, is_withdraw=False):
        if self.ROUTE_FAMILY == RF_L2_EVPN:
            vrf_nlri = self._nlri
        else:
            vrf_nlri = self.NLRI_CLASS(self._nlri.prefix)
        pathattrs = None
        if not is_withdraw:
            pathattrs = self.pathattr_map
        vrf_path = self.VRF_PATH_CLASS(puid=self.VRF_PATH_CLASS.create_puid(self._nlri.route_dist, self._nlri.prefix), source=self.source, nlri=vrf_nlri, src_ver_num=self.source_version_num, pattrs=pathattrs, nexthop=self.nexthop, is_withdraw=is_withdraw, label_list=self._nlri.label_list)
        return vrf_path