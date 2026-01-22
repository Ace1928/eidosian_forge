from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def _get_additional_args(self):
    additional_args = dict()
    if self.module.params.get('isolation_method'):
        additional_args.update(dict(isolationmethod=self.module.params.get('isolation_method')))
    if self.module.params.get('vlan'):
        additional_args.update(dict(vlan=self.module.params.get('vlan')))
    additional_args.update(self._get_label_args())
    return additional_args