from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def _freeze_iam_role(self, iam_role_arn):
    if hasattr(self, 'ansible_name'):
        role_session_name = f'ansible_aws_{self.ansible_name}_dynamic_inventory'
    else:
        role_session_name = 'ansible_aws_dynamic_inventory'
    assume_params = {'RoleArn': iam_role_arn, 'RoleSessionName': role_session_name}
    try:
        sts = self.client('sts')
        assumed_role = sts.assume_role(**assume_params)
    except AnsibleBotocoreError as e:
        self.fail_aws(f'Unable to assume role {iam_role_arn}', exception=e)
    credentials = assumed_role.get('Credentials')
    if not credentials:
        self.fail_aws(f'Unable to assume role {iam_role_arn}')
    self._frozen_credentials = {'profile_name': None, 'aws_access_key_id': credentials.get('AccessKeyId'), 'aws_secret_access_key': credentials.get('SecretAccessKey'), 'aws_session_token': credentials.get('SessionToken')}