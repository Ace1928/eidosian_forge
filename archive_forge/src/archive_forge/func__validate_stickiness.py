from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _validate_stickiness(self, stickiness):
    problem_found = False
    if not stickiness:
        return problem_found
    if not stickiness['enabled']:
        return problem_found
    if stickiness['type'] == 'application':
        if not stickiness.get('cookie'):
            problem_found = True
            self.module.fail_json(msg='cookie must be specified when stickiness type is "application"', stickiness=stickiness)
        if stickiness.get('expiration'):
            self.warn(msg='expiration is ignored when stickiness type is "application"')
    if stickiness['type'] == 'loadbalancer':
        if stickiness.get('cookie'):
            self.warn(msg='cookie is ignored when stickiness type is "loadbalancer"')
    return problem_found