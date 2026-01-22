from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
def absent_image_store(self):
    image_store = self.get_image_store()
    if image_store:
        self.result['changed'] = True
        if not self.module.check_mode:
            args = {'id': image_store.get('id')}
            self.query_api('deleteImageStore', **args)
    return image_store