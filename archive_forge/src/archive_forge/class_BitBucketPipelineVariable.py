from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, _load_params
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
class BitBucketPipelineVariable(AnsibleModule):

    def __init__(self, *args, **kwargs):
        params = _load_params() or {}
        if params.get('secured'):
            kwargs['argument_spec']['value'].update({'no_log': True})
        super(BitBucketPipelineVariable, self).__init__(*args, **kwargs)