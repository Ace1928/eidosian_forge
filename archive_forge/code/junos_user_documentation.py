from __future__ import absolute_import, division, print_function
from copy import deepcopy
from functools import partial
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
main entry point for module execution