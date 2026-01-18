from ansible.errors import AnsibleLookupError
from ansible.module_utils._text import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.lookup import AWSLookupBase

        :arg terms: a list of lookups to run.
            e.g. ['parameter_name', 'parameter_name_too' ]
        :kwarg variables: ansible variables active at the time of the lookup
        :returns: A list of parameter values or a list of dictionaries if bypath=True.
        