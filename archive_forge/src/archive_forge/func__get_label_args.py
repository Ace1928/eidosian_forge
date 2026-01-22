from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def _get_label_args(self):
    label_args = dict()
    if self.module.params.get('hyperv_networklabel'):
        label_args.update(dict(hypervnetworklabel=self.module.params.get('hyperv_networklabel')))
    if self.module.params.get('kvm_networklabel'):
        label_args.update(dict(kvmnetworklabel=self.module.params.get('kvm_networklabel')))
    if self.module.params.get('ovm3_networklabel'):
        label_args.update(dict(ovm3networklabel=self.module.params.get('ovm3_networklabel')))
    if self.module.params.get('vmware_networklabel'):
        label_args.update(dict(vmwarenetworklabel=self.module.params.get('vmware_networklabel')))
    return label_args