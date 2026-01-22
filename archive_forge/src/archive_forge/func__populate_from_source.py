from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _populate_from_source(self, source_data):
    hostvars = source_data.pop('_meta', {}).get('hostvars', {})
    for group in source_data:
        if group == 'all':
            continue
        self.inventory.add_group(group)
        hosts = source_data[group].get('hosts', [])
        for host in hosts:
            self._populate_host_vars([host], hostvars.get(host, {}), group)
        self.inventory.add_child('all', group)