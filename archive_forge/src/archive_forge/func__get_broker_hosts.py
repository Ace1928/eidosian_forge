from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_broker_hosts(self, connection, strict):

    def _boto3_paginate_wrapper(func, *args, **kwargs):
        results = []
        try:
            results = func(*args, **kwargs)
            results = results['BrokerSummaries']
            _add_details_to_hosts(connection, results, strict)
        except is_boto3_error_code('AccessDenied') as e:
            if not strict:
                results = []
            else:
                raise AnsibleError(f'Failed to query MQ: {to_native(e)}')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            raise AnsibleError(f'Failed to query MQ: {to_native(e)}')
        return results
    return _boto3_paginate_wrapper