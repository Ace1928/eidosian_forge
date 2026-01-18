from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.netconf.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.utils.data import (
main entry point for module execution