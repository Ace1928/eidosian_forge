from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def update_tls_hostname(result, old_behavior=False, deprecate_function=None, uses_tls=True):
    if result['tls_hostname'] is None:
        parsed_url = urlparse(result['docker_host'])
        result['tls_hostname'] = parsed_url.netloc.rsplit(':', 1)[0]