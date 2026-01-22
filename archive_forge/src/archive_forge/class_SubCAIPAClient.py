from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class SubCAIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(SubCAIPAClient, self).__init__(module, host, port, protocol)

    def subca_find(self, subca_name):
        return self._post_json(method='ca_find', name=subca_name, item=None)

    def subca_add(self, subca_name=None, subject_dn=None, details=None):
        item = dict(ipacasubjectdn=subject_dn)
        subca_desc = details.get('description', None)
        if subca_desc is not None:
            item.update(description=subca_desc)
        return self._post_json(method='ca_add', name=subca_name, item=item)

    def subca_mod(self, subca_name=None, diff=None, details=None):
        item = get_subca_dict(details)
        for change in diff:
            update_detail = dict()
            if item[change] is not None:
                update_detail.update(setattr='{0}={1}'.format(change, item[change]))
                self._post_json(method='ca_mod', name=subca_name, item=update_detail)

    def subca_del(self, subca_name=None):
        return self._post_json(method='ca_del', name=subca_name)

    def subca_disable(self, subca_name=None):
        return self._post_json(method='ca_disable', name=subca_name)

    def subca_enable(self, subca_name=None):
        return self._post_json(method='ca_enable', name=subca_name)