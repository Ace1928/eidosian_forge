from __future__ import absolute_import, division, print_function
import re
import shlex
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_lines
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
entry point for module execution