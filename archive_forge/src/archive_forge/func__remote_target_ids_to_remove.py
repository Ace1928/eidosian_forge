import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _remote_target_ids_to_remove(self):
    """Returns a list of targets that need to be removed remotely"""
    target_ids = [t['id'] for t in self.targets]
    remote_targets = self.rule.list_targets()
    return [rt['id'] for rt in remote_targets if rt['id'] not in target_ids]