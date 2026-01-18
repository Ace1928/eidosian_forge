import json
from ansible.errors import AnsibleLookupError
from ansible.module_utils._text import to_native
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.plugin_utils.lookup import AWSLookupBase

        :arg terms: a list of lookups to run.
            e.g. ['example_secret_name', 'example_secret_too' ]
        :variables: ansible variables active at the time of the lookup
        :returns: A list of parameter values or a list of dictionaries if bypath=True.
        