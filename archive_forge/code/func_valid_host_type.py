from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
@property
def valid_host_type(self):
    host_types = None
    try:
        rc, host_types = self.request('storage-systems/%s/host-types' % self.ssid)
    except Exception as err:
        self.module.fail_json(msg='Failed to get host types. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    try:
        match = list(filter(lambda host_type: host_type['index'] == self.host_type_index, host_types))[0]
        return True
    except IndexError:
        self.module.fail_json(msg='There is no host type with index %s' % self.host_type_index)