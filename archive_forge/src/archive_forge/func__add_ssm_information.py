import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _add_ssm_information(self, connection, instances):
    instance_ids = [x['InstanceId'] for x in instances]
    result = self._get_multiple_ssm_inventories(connection, instance_ids)
    for entity in result.get('Entities', []):
        for x in instances:
            if x['InstanceId'] == entity['Id']:
                content = entity.get('Data', {}).get('AWS:InstanceInformation', {}).get('Content', [])
                if content:
                    x['SsmInventory'] = content[0]
                break