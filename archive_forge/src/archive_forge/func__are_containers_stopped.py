from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
def _are_containers_stopped(self):
    for container in self.list_containers_raw():
        if container['State'] not in ('created', 'exited', 'stopped', 'killed'):
            return False
    return True