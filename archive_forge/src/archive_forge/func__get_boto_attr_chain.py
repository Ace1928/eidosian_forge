import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_boto_attr_chain(filter_name, instance):
    """
    :param filter_name: The filter
    :param instance: instance dict returned by boto3 ec2 describe_instances()
    """
    allowed_filters = sorted(list(instance_data_filter_to_boto_attr.keys()) + list(instance_meta_filter_to_boto_attr.keys()))
    if filter_name not in allowed_filters:
        return filter_name
    if filter_name in instance_data_filter_to_boto_attr:
        boto_attr_list = instance_data_filter_to_boto_attr[filter_name]
    else:
        boto_attr_list = instance_meta_filter_to_boto_attr[filter_name]
    instance_value = instance
    for attribute in boto_attr_list:
        instance_value = _compile_values(instance_value, attribute)
    return instance_value