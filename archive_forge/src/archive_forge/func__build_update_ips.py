from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
def _build_update_ips(self, server):
    auto_ip = self.params['auto_ip']
    floating_ips = self.params['floating_ips']
    floating_ip_pools = self.params['floating_ip_pools']
    if not (auto_ip or floating_ips or floating_ip_pools):
        return {}
    ips = [interface_spec['addr'] for v in server['addresses'].values() for interface_spec in v if interface_spec.get('OS-EXT-IPS:type', None) == 'floating']
    if auto_ip and ips and (not floating_ip_pools) and (not floating_ips):
        return {}
    if not ips:
        return dict(ips=dict(auto_ip=auto_ip, ips=floating_ips, ip_pool=floating_ip_pools))
    if auto_ip or not floating_ips:
        return {}
    update = {}
    add_ips = [ip for ip in floating_ips if ip not in ips]
    if add_ips:
        update['add_ips'] = add_ips
    remove_ips = [ip for ip in ips if ip not in floating_ips]
    if remove_ips:
        update['remove_ips'] = remove_ips