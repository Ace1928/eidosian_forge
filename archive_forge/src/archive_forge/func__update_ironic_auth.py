from ansible.module_utils.basic import AnsibleModule
from ansible_collections.openstack.cloud.plugins.module_utils.openstack import openstack_full_argument_spec
def _update_ironic_auth(self):
    """Validate and update authentication parameters for ironic."""
    if self.params['auth_type'] in [None, 'None', 'none'] and self.params['ironic_url'] is None and (not self.params['cloud']) and (not (self.params['auth'] and self.params['auth'].get('endpoint'))):
        self.fail_json(msg='Authentication appears to be disabled, Please define either ironic_url, or cloud, or auth.endpoint')
    if self.params['ironic_url'] and self.params['auth_type'] in [None, 'None', 'none'] and (not (self.params['auth'] and self.params['auth'].get('endpoint'))):
        self.params['auth'] = dict(endpoint=self.params['ironic_url'])