from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _rename_profile(self):
    url = '/1.0/profiles/{0}'.format(self.name)
    if self.project:
        url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
    config = {'name': self.new_name}
    self.client.do('POST', url, config)
    self.actions.append('rename')
    self.name = self.new_name