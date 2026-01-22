from __future__ import absolute_import, division, print_function
from ipaddress import ip_interface
from ansible.module_utils._text import to_text
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (
def _get_new_available_prefix(self, data, endpoint_name):
    if not self.nb_object:
        self.result['changed'] = False
        self.result['msg'] = 'Parent prefix does not exist - %s' % data['parent']
    elif self.nb_object.available_prefixes.list():
        if self.check_mode:
            self.result['changed'] = True
            self.result['msg'] = 'New prefix created within %s' % data['parent']
            self.module.exit_json(**self.result)
        self.nb_object, diff = self._create_netbox_object(self.nb_object.available_prefixes, data)
        self.nb_object = self.nb_object.serialize()
        self.result['changed'] = True
        self.result['msg'] = '%s %s created' % (endpoint_name, self.nb_object['prefix'])
        self.result['diff'] = diff
    else:
        self.result['changed'] = False
        self.result['msg'] = 'No available prefixes within %s' % data['parent']