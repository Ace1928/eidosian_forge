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
Check if the available botocore version is greater than or equal to a desired version.

    Usage:
        if not module.botocore_at_least('1.2.3'):
            module.fail_json(msg='The Serverless Elastic Load Compute Service is not in botocore before v1.2.3')
        if not module.botocore_at_least('1.5.3'):
            module.warn('Botocore did not include waiters for Service X before 1.5.3. '
                        'To wait until Service X resources are fully available, update botocore.')
    