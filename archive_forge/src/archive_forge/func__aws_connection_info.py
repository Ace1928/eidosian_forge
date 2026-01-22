import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def _aws_connection_info(params):
    endpoint_url = params.get('endpoint_url')
    access_key = params.get('access_key')
    secret_key = params.get('secret_key')
    session_token = params.get('session_token')
    region = _aws_region(params)
    profile_name = params.get('profile')
    validate_certs = params.get('validate_certs')
    ca_bundle = params.get('aws_ca_bundle')
    config = params.get('aws_config')
    if profile_name and (access_key or secret_key or session_token):
        raise AnsibleBotocoreError(message='Passing both a profile and access tokens is not supported.')
    if not access_key:
        access_key = None
    if not secret_key:
        secret_key = None
    if not session_token:
        session_token = None
    if profile_name:
        boto_params = dict(aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, profile_name=profile_name)
    else:
        boto_params = dict(aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=session_token)
    if validate_certs and ca_bundle:
        boto_params['verify'] = ca_bundle
    else:
        boto_params['verify'] = validate_certs
    if config is not None:
        boto_params['aws_config'] = botocore.config.Config(**config)
    for param, value in boto_params.items():
        if isinstance(value, binary_type):
            boto_params[param] = text_type(value, 'utf-8', 'strict')
    return (region, endpoint_url, boto_params)