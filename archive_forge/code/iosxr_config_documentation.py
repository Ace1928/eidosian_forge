from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
main entry point for module execution