from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _status_check(name, channel, installed):
    match = [c for n, c in installed if n == name]
    if not match:
        return Snap.NOT_INSTALLED
    if channel and match[0] not in (channel, 'latest/{0}'.format(channel)):
        return Snap.CHANNEL_MISMATCH
    else:
        return Snap.INSTALLED