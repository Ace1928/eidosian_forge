from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_indexes_active
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_exists
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_not_exists
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _generate_attribute_map():
    """
    Builds a map of Key Names to Type
    """
    attributes = dict()
    for index in (module.params, *module.params.get('indexes')):
        for t in ['hash', 'range']:
            key_name = index.get(t + '_key_name')
            if not key_name:
                continue
            key_type = index.get(t + '_key_type') or DYNAMO_TYPE_DEFAULT
            _type = _long_type_to_short(key_type)
            if key_name in attributes:
                if _type != attributes[key_name]:
                    module.fail_json(msg='Conflicting attribute type', type_1=_type, type_2=attributes[key_name], key_name=key_name)
            else:
                attributes[key_name] = _type
    return attributes