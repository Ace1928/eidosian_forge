from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def update_host_group(self):
    """Update host group."""
    data = {'name': self.name, 'hosts': self.hosts}
    desired_host_ids = self.hosts
    for host in self.current_hosts_in_host_group:
        if host not in desired_host_ids:
            self.unassign_hosts([host])
    update_response = None
    try:
        rc, update_response = self.request('storage-systems/%s/host-groups/%s' % (self.ssid, self.current_host_group['id']), method='POST', data=data)
    except Exception as error:
        self.module.fail_json(msg='Failed to create host group. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    return update_response