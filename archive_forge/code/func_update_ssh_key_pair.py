from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def update_ssh_key_pair(module, bitbucket):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['ssh-key-pair'].format(workspace=module.params['workspace'], repo_slug=module.params['repository']), method='PUT', data={'private_key': module.params['private_key'], 'public_key': module.params['public_key']})
    if info['status'] == 404:
        module.fail_json(msg=error_messages['invalid_params'])
    if info['status'] != 200:
        module.fail_json(msg='Failed to create or update pipeline ssh key pair : {0}'.format(info))