from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
def _update_ips(self, server, update):
    args = dict(((k, self.params[k]) for k in ['wait', 'timeout']))
    ips = update.get('ips')
    if ips:
        server = self.conn.add_ips_to_server(server, **ips, **args)
    add_ips = update.get('add_ips')
    if add_ips:
        server = self.conn.add_ip_list(server, add_ips, **args)
    remove_ips = update.get('remove_ips')
    if remove_ips:
        for ip in remove_ips:
            ip_id = self.conn.network.find_ip(name_or_id=ip, ignore_missing=False).id
            self.conn.detach_ip_from_server(server_id=server.id, floating_ip_id=ip_id)
    return server