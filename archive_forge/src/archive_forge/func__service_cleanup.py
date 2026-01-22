import typing as ty
from openstack import exceptions
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import (
from openstack.network.v2 import availability_zone
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import extension
from openstack.network.v2 import firewall_group as _firewall_group
from openstack.network.v2 import firewall_policy as _firewall_policy
from openstack.network.v2 import firewall_rule as _firewall_rule
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import floating_ip as _floating_ip
from openstack.network.v2 import health_monitor as _health_monitor
from openstack.network.v2 import l3_conntrack_helper as _l3_conntrack_helper
from openstack.network.v2 import listener as _listener
from openstack.network.v2 import load_balancer as _load_balancer
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import metering_label as _metering_label
from openstack.network.v2 import metering_label_rule as _metering_label_rule
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability
from openstack.network.v2 import (
from openstack.network.v2 import pool as _pool
from openstack.network.v2 import pool_member as _pool_member
from openstack.network.v2 import port as _port
from openstack.network.v2 import port_forwarding as _port_forwarding
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import qos_policy as _qos_policy
from openstack.network.v2 import qos_rule_type as _qos_rule_type
from openstack.network.v2 import quota as _quota
from openstack.network.v2 import rbac_policy as _rbac_policy
from openstack.network.v2 import router as _router
from openstack.network.v2 import security_group as _security_group
from openstack.network.v2 import security_group_rule as _security_group_rule
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _service_profile
from openstack.network.v2 import service_provider as _service_provider
from openstack.network.v2 import sfc_flow_classifier as _sfc_flow_classifier
from openstack.network.v2 import sfc_port_chain as _sfc_port_chain
from openstack.network.v2 import sfc_port_pair as _sfc_port_pair
from openstack.network.v2 import sfc_port_pair_group as _sfc_port_pair_group
from openstack.network.v2 import sfc_service_graph as _sfc_sservice_graph
from openstack.network.v2 import subnet as _subnet
from openstack.network.v2 import subnet_pool as _subnet_pool
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.network.v2 import trunk as _trunk
from openstack.network.v2 import vpn_endpoint_group as _vpn_endpoint_group
from openstack.network.v2 import vpn_ike_policy as _ike_policy
from openstack.network.v2 import vpn_ipsec_policy as _ipsec_policy
from openstack.network.v2 import (
from openstack.network.v2 import vpn_service as _vpn_service
from openstack import proxy
from openstack import resource
def _service_cleanup(self, dry_run=True, client_status_queue=None, identified_resources=None, filters=None, resource_evaluation_fn=None, skip_resources=None):
    project_id = self.get_project_id()
    if not self.should_skip_resource_cleanup('floating_ip', skip_resources):
        for obj in self.ips(project_id=project_id):
            self._service_cleanup_del_res(self.delete_ip, obj, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=filters, resource_evaluation_fn=fip_cleanup_evaluation)
    if not self.should_skip_resource_cleanup('security_group', skip_resources):
        for obj in self.security_groups(project_id=project_id):
            if obj.name != 'default':
                self._service_cleanup_del_res(self.delete_security_group, obj, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=filters, resource_evaluation_fn=resource_evaluation_fn)
    if not (self.should_skip_resource_cleanup('network', skip_resources) or self.should_skip_resource_cleanup('router', skip_resources) or self.should_skip_resource_cleanup('port', skip_resources) or self.should_skip_resource_cleanup('subnet', skip_resources)):
        for net in self.networks(project_id=project_id):
            network_has_ports_allocated = False
            router_if = list()
            for port in self.ports(project_id=project_id, network_id=net.id):
                self.log.debug('Looking at port %s' % port)
                if port.device_owner in ['network:router_interface', 'network:router_interface_distributed', 'network:ha_router_replicated_interface']:
                    router_if.append(port)
                elif port.device_owner == 'network:dhcp':
                    continue
                elif port.device_owner is None or port.device_owner == '':
                    continue
                elif identified_resources and port.device_id not in identified_resources:
                    network_has_ports_allocated = True
            if network_has_ports_allocated:
                continue
            self.log.debug('Network %s should be deleted' % net)
            network_must_be_deleted = self._service_cleanup_del_res(self.delete_network, net, dry_run=True, client_status_queue=None, identified_resources=None, filters=filters, resource_evaluation_fn=resource_evaluation_fn)
            if not network_must_be_deleted:
                continue
            for port in router_if:
                if client_status_queue:
                    client_status_queue.put(port)
                router = self.get_router(port.device_id)
                if not dry_run:
                    if len(router.routes) > 0:
                        try:
                            self.remove_extra_routes_from_router(router, {'router': {'routes': router.routes}})
                        except exceptions.SDKException:
                            self.log.error(f'Cannot delete routes {router.routes} from router {router}')
                    try:
                        self.remove_interface_from_router(router=port.device_id, port_id=port.id)
                    except exceptions.SDKException:
                        self.log.error('Cannot delete object %s' % obj)
                self._service_cleanup_del_res(self.delete_router, router, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=None, resource_evaluation_fn=None)
            for port in self.ports(project_id=project_id, network_id=net.id):
                if port.device_owner is None or port.device_owner == '':
                    self._service_cleanup_del_res(self.delete_port, port, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=None, resource_evaluation_fn=None)
            for obj in self.subnets(project_id=project_id, network_id=net.id):
                self._service_cleanup_del_res(self.delete_subnet, obj, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=None, resource_evaluation_fn=None)
            self._service_cleanup_del_res(self.delete_network, net, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=None, resource_evaluation_fn=None)
    else:
        self.log.debug('Skipping cleanup of networks, routers, ports and subnets as those resources require all of them to be cleaned uptogether, but at least one should be kept')
    if not self.should_skip_resource_cleanup('router', skip_resources):
        for obj in self.routers():
            ports = list(self.ports(device_id=obj.id))
            if len(ports) == 0:
                self._service_cleanup_del_res(self.delete_router, obj, dry_run=dry_run, client_status_queue=client_status_queue, identified_resources=identified_resources, filters=None, resource_evaluation_fn=None)