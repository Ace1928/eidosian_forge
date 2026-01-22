from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _add_tags_for_rds_hosts(connection, hosts, strict):
    for host in hosts:
        if 'DBInstanceArn' in host:
            resource_arn = host['DBInstanceArn']
        else:
            resource_arn = host['DBClusterArn']
        try:
            tags = connection.list_tags_for_resource(ResourceName=resource_arn)['TagList']
        except is_boto3_error_code('AccessDenied') as e:
            if not strict:
                tags = []
            else:
                raise e
        host['Tags'] = tags