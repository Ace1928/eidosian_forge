from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
Execute a HTTP request and handle common things like rate limiting.