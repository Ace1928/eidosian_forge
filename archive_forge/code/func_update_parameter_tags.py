import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_parameter_tags(client, module, parameter_name, supplied_tags):
    changed = False
    response = {}
    if supplied_tags is None:
        return (False, response)
    current_tags = get_parameter_tags(client, module, parameter_name)
    tags_to_add, tags_to_remove = compare_aws_tags(current_tags, supplied_tags, module.params.get('purge_tags'))
    if tags_to_add:
        if module.check_mode:
            return (True, response)
        response = tag_parameter(client, module, parameter_name, ansible_dict_to_boto3_tag_list(tags_to_add))
        changed = True
    if tags_to_remove:
        if module.check_mode:
            return (True, response)
        response = untag_parameter(client, module, parameter_name, tags_to_remove)
        changed = True
    return (changed, response)