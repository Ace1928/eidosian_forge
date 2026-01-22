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
def _paginated_query(client, paginator_name, **params):
    paginator = client.get_paginator(paginator_name)
    result = paginator.paginate(**params).build_full_result()
    return result