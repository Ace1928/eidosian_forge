import json
from ansible.errors import AnsibleLookupError
from ansible.module_utils._text import to_native
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.plugin_utils.lookup import AWSLookupBase
def _list_secrets(client, term):
    paginator = client.get_paginator('list_secrets')
    return paginator.paginate(Filters=[{'Key': 'name', 'Values': [term]}])