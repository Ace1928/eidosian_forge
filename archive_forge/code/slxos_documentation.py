from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection
Apply a list of commands to a device.

    Given a list of commands apply them to the device to modify the
    configuration in bulk.

    Args:
        module: A valid AnsibleModule instance.
        commands: Iterable of command strings.

    Returns:
        None
    