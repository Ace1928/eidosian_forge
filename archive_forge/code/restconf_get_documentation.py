from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.restconf import restconf
from ansible_collections.ansible.netcommon.plugins.module_utils.utils.data import xml_to_dict
entry point for module execution