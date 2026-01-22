import abc
import logging
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGPEncapsulationExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsiLabelExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsImportRTExtendedCommunity
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import PmsiTunnelIdIngressReplication
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.safi import (
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.utils.bgp import create_rt_extended_community
from os_ken.services.protocols.bgp.utils.stats import LOCAL_ROUTES
from os_ken.services.protocols.bgp.utils.stats import REMOTE_ROUTES
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_ID
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_NAME
class VrfTable(Table, metaclass=abc.ABCMeta):
    """Virtual Routing and Forwarding information base.
     Keeps destination imported to given vrf in represents.
     """
    ROUTE_FAMILY = None
    VPN_ROUTE_FAMILY = None
    NLRI_CLASS = None
    VRF_PATH_CLASS = None
    VRF_DEST_CLASS = None

    def __init__(self, vrf_conf, core_service, signal_bus):
        Table.__init__(self, vrf_conf.route_dist, core_service, signal_bus)
        self._vrf_conf = vrf_conf
        self._import_maps = []
        self.init_import_maps(vrf_conf.import_maps)

    def init_import_maps(self, import_maps):
        LOG.debug('Initializing import maps (%s) for %r', import_maps, self)
        del self._import_maps[:]
        importmap_manager = self._core_service.importmap_manager
        for name in import_maps:
            import_map = importmap_manager.get_import_map_by_name(name)
            if import_map is None:
                raise KeyError('No import map with name %s' % name)
            self._import_maps.append(import_map)

    @property
    def import_rts(self):
        return self._vrf_conf.import_rts

    @property
    def vrf_conf(self):
        return self._vrf_conf

    def _table_key(self, nlri):
        """Return a key that will uniquely identify this NLRI inside
        this table.
        """
        return nlri.prefix

    def _create_dest(self, nlri):
        return self.VRF_DEST_CLASS(self, nlri)

    def append_import_map(self, import_map):
        self._import_maps.append(import_map)

    def remove_import_map(self, import_map):
        self._import_maps.remove(import_map)

    def get_stats_summary_dict(self):
        """Returns count of local and remote paths."""
        remote_route_count = 0
        local_route_count = 0
        for dest in self.values():
            for path in dest.known_path_list:
                if hasattr(path.source, 'version_num') or path.source == VPN_TABLE:
                    remote_route_count += 1
                else:
                    local_route_count += 1
        return {RESOURCE_ID: self._vrf_conf.id, RESOURCE_NAME: self._vrf_conf.name, REMOTE_ROUTES: remote_route_count, LOCAL_ROUTES: local_route_count}

    def import_vpn_paths_from_table(self, vpn_table, import_rts=None):
        for vpn_dest in vpn_table.values():
            vpn_path = vpn_dest.best_path
            if not vpn_path:
                continue
            if import_rts is None:
                import_rts = set(self.import_rts)
            else:
                import_rts = set(import_rts)
            path_rts = vpn_path.get_rts()
            if import_rts.intersection(path_rts):
                self.import_vpn_path(vpn_path)

    def import_vpn_path(self, vpn_path):
        """Imports `vpnv(4|6)_path` into `vrf(4|6)_table` or `evpn_path`
        into vrfevpn_table`.

        :Parameters:
            - `vpn_path`: (Path) VPN path that will be cloned and imported
            into VRF.
        Note: Does not do any checking if this import is valid.
        """
        assert vpn_path.route_family == self.VPN_ROUTE_FAMILY
        source = vpn_path.source
        if not source:
            source = VRF_TABLE
        if self.VPN_ROUTE_FAMILY == RF_L2_EVPN:
            vrf_nlri = vpn_path.nlri
        elif self.ROUTE_FAMILY.safi in [IP_FLOWSPEC, VPN_FLOWSPEC]:
            vrf_nlri = self.NLRI_CLASS(rules=vpn_path.nlri.rules)
        else:
            ip, masklen = vpn_path.nlri.prefix.split('/')
            vrf_nlri = self.NLRI_CLASS(length=int(masklen), addr=ip)
        vrf_path = self.VRF_PATH_CLASS(puid=self.VRF_PATH_CLASS.create_puid(vpn_path.nlri.route_dist, vpn_path.nlri.prefix), source=source, nlri=vrf_nlri, src_ver_num=vpn_path.source_version_num, pattrs=vpn_path.pathattr_map, nexthop=vpn_path.nexthop, is_withdraw=vpn_path.is_withdraw, label_list=getattr(vpn_path.nlri, 'label_list', None))
        if self._is_vrf_path_already_in_table(vrf_path):
            return None
        if self._is_vrf_path_filtered_out_by_import_maps(vrf_path):
            return None
        else:
            vrf_dest = self.insert(vrf_path)
            self._signal_bus.dest_changed(vrf_dest)

    def _is_vrf_path_filtered_out_by_import_maps(self, vrf_path):
        for import_map in self._import_maps:
            if import_map.match(vrf_path):
                return True
        return False

    def _is_vrf_path_already_in_table(self, vrf_path):
        dest = self._get_dest(vrf_path.nlri)
        if dest is None:
            return False
        return vrf_path in dest.known_path_list

    def apply_import_maps(self):
        changed_dests = []
        for dest in self.values():
            assert isinstance(dest, VrfDest)
            for import_map in self._import_maps:
                for path in dest.known_path_list:
                    if import_map.match(path):
                        dest.withdraw_path(path)
                        changed_dests.append(dest)
        return changed_dests

    def insert_vrf_path(self, nlri, next_hop=None, gen_lbl=False, is_withdraw=False, **kwargs):
        assert nlri
        pattrs = None
        label_list = []
        vrf_conf = self.vrf_conf
        if not is_withdraw:
            table_manager = self._core_service.table_manager
            if gen_lbl and next_hop:
                label_key = (vrf_conf.route_dist, next_hop)
                nh_label = table_manager.get_nexthop_label(label_key)
                if not nh_label:
                    nh_label = table_manager.get_next_vpnv4_label()
                    table_manager.set_nexthop_label(label_key, nh_label)
                label_list.append(nh_label)
            elif gen_lbl:
                label_list.append(table_manager.get_next_vpnv4_label())
            if gen_lbl and isinstance(nlri, EvpnMacIPAdvertisementNLRI):
                nlri.mpls_labels = label_list[:2]
            elif gen_lbl and isinstance(nlri, EvpnIpPrefixNLRI):
                nlri.mpls_label = label_list[0]
            pattrs = OrderedDict()
            from os_ken.services.protocols.bgp.core import EXPECTED_ORIGIN
            pattrs[BGP_ATTR_TYPE_ORIGIN] = BGPPathAttributeOrigin(EXPECTED_ORIGIN)
            pattrs[BGP_ATTR_TYPE_AS_PATH] = BGPPathAttributeAsPath([])
            communities = []
            if isinstance(nlri, EvpnEthernetSegmentNLRI):
                subtype = 2
                es_import = nlri.esi.mac_addr
                communities.append(BGPEvpnEsImportRTExtendedCommunity(subtype=subtype, es_import=es_import))
            for rt in vrf_conf.export_rts:
                communities.append(create_rt_extended_community(rt, 2))
            for soo in vrf_conf.soo_list:
                communities.append(create_rt_extended_community(soo, 3))
            tunnel_type = kwargs.get('tunnel_type', None)
            if tunnel_type:
                communities.append(BGPEncapsulationExtendedCommunity.from_str(tunnel_type))
            redundancy_mode = kwargs.get('redundancy_mode', None)
            if redundancy_mode is not None:
                subtype = 1
                flags = 0
                from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
                if redundancy_mode == REDUNDANCY_MODE_SINGLE_ACTIVE:
                    flags |= BGPEvpnEsiLabelExtendedCommunity.SINGLE_ACTIVE_BIT
                vni = kwargs.get('vni', None)
                if vni is not None:
                    communities.append(BGPEvpnEsiLabelExtendedCommunity(subtype=subtype, flags=flags, vni=vni))
                else:
                    communities.append(BGPEvpnEsiLabelExtendedCommunity(subtype=subtype, flags=flags, mpls_label=label_list[0]))
            mac_mobility_seq = kwargs.get('mac_mobility', None)
            if mac_mobility_seq is not None:
                from os_ken.lib.packet.bgp import BGPEvpnMacMobilityExtendedCommunity
                is_static = mac_mobility_seq == -1
                communities.append(BGPEvpnMacMobilityExtendedCommunity(subtype=0, flags=1 if is_static else 0, sequence_number=mac_mobility_seq if not is_static else 0))
            pattrs[BGP_ATTR_TYPE_EXTENDED_COMMUNITIES] = BGPPathAttributeExtendedCommunities(communities=communities)
            if vrf_conf.multi_exit_disc:
                pattrs[BGP_ATTR_TYPE_MULTI_EXIT_DISC] = BGPPathAttributeMultiExitDisc(vrf_conf.multi_exit_disc)
            pmsi_tunnel_type = kwargs.get('pmsi_tunnel_type', None)
            if pmsi_tunnel_type is not None:
                from os_ken.services.protocols.bgp.api.prefix import PMSI_TYPE_INGRESS_REP
                if pmsi_tunnel_type == PMSI_TYPE_INGRESS_REP:
                    vtep = kwargs.get('tunnel_endpoint_ip', self._core_service.router_id)
                    tunnel_id = PmsiTunnelIdIngressReplication(tunnel_endpoint_ip=vtep)
                else:
                    tunnel_id = None
                pattrs[BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE] = BGPPathAttributePmsiTunnel(pmsi_flags=0, tunnel_type=pmsi_tunnel_type, tunnel_id=tunnel_id, vni=kwargs.get('vni', None))
        puid = self.VRF_PATH_CLASS.create_puid(vrf_conf.route_dist, nlri.prefix)
        path = self.VRF_PATH_CLASS(puid, None, nlri, 0, pattrs=pattrs, nexthop=next_hop, label_list=label_list, is_withdraw=is_withdraw)
        eff_dest = self.insert(path)
        self._signal_bus.dest_changed(eff_dest)
        return label_list

    def clean_uninteresting_paths(self, interested_rts=None):
        if interested_rts is None:
            interested_rts = set(self.vrf_conf.import_rts)
        return super(VrfTable, self).clean_uninteresting_paths(interested_rts)