from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class HostIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(HostIPAClient, self).__init__(module, host, port, protocol)

    def host_show(self, name):
        return self._post_json(method='host_show', name=name)

    def host_find(self, name):
        return self._post_json(method='host_find', name=None, item={'all': True, 'fqdn': name})

    def host_add(self, name, host):
        return self._post_json(method='host_add', name=name, item=host)

    def host_mod(self, name, host):
        return self._post_json(method='host_mod', name=name, item=host)

    def host_del(self, name, update_dns):
        return self._post_json(method='host_del', name=name, item={'updatedns': update_dns})

    def host_disable(self, name):
        return self._post_json(method='host_disable', name=name)