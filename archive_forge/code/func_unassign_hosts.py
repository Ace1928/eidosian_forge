from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def unassign_hosts(self, host_list=None):
    """Unassign hosts from host group."""
    if host_list is None:
        host_list = self.current_host_group['hosts']
    for host_id in host_list:
        try:
            rc, resp = self.request('storage-systems/%s/hosts/%s/move' % (self.ssid, host_id), method='POST', data={'group': '0000000000000000000000000000000000000000'})
        except Exception as error:
            self.module.fail_json(msg='Failed to unassign hosts from host group. Array id [%s].  Host id [%s].  Error[%s].' % (self.ssid, host_id, to_native(error)))