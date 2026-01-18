from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible_collections.community.routeros.plugins.module_utils.version import LooseVersion
from ansible.module_utils.connection import Connection, ConnectionError
Determines whether adding a leading space to the command is needed
    to workaround prompt bug in 6.49 <= ROS < 7.2