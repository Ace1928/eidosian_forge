from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (Config, HwcClientException,
import re
def _push_policy_convert_from_response(value):
    return {0: 'the message sending fails and is cached in the queue', 1: 'the failed message is discarded'}.get(int(value))