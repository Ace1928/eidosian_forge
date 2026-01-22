import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _sync_targets(self):
    """Syncs local targets with AWS"""
    target_ids_to_remove = self._remote_target_ids_to_remove()
    if target_ids_to_remove:
        self.rule.remove_targets(target_ids_to_remove)
    targets_to_put = self._targets_to_put()
    if targets_to_put:
        self.rule.put_targets(targets_to_put)