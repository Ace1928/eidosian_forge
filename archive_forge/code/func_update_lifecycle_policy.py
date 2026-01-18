from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_lifecycle_policy(self, name, transition_to_ia):
    """
        Update filesystem with new lifecycle policy.
        """
    changed = False
    state = self.get_file_system_state(name)
    if state in [self.STATE_AVAILABLE, self.STATE_CREATING]:
        fs_id = self.get_file_system_id(name)
        current_policies = self.connection.describe_lifecycle_configuration(FileSystemId=fs_id)
        if transition_to_ia == 'None':
            LifecyclePolicies = []
        else:
            LifecyclePolicies = [{'TransitionToIA': 'AFTER_' + transition_to_ia + '_DAYS'}]
        if current_policies.get('LifecyclePolicies') != LifecyclePolicies:
            response = self.connection.put_lifecycle_configuration(FileSystemId=fs_id, LifecyclePolicies=LifecyclePolicies)
            changed = True
    return changed