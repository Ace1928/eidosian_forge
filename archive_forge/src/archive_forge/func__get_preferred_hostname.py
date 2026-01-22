import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_preferred_hostname(self, instance, hostnames):
    """
        :param instance: an instance dict returned by boto3 ec2 describe_instances()
        :param hostnames: a list of hostname destination variables in order of preference
        :return the preferred identifer for the host
        """
    if not hostnames:
        hostnames = ['dns-name', 'private-dns-name']
    hostname = None
    for preference in hostnames:
        if isinstance(preference, dict):
            if 'name' not in preference:
                self.fail_aws("A 'name' key must be defined in a hostnames dictionary.")
            hostname = self._get_preferred_hostname(instance, [preference['name']])
            hostname_from_prefix = None
            if 'prefix' in preference:
                hostname_from_prefix = self._get_preferred_hostname(instance, [preference['prefix']])
            separator = preference.get('separator', '_')
            if hostname and hostname_from_prefix and ('prefix' in preference):
                hostname = hostname_from_prefix + separator + hostname
        elif preference.startswith('tag:'):
            tags = _get_tag_hostname(preference, instance)
            hostname = tags[0] if tags else None
        else:
            hostname = _get_boto_attr_chain(preference, instance)
        if hostname:
            break
    if hostname:
        return self._sanitize_hostname(hostname)