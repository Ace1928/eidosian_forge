from __future__ import (absolute_import, division, print_function)
import traceback
import json
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import (
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _get_stack_slugs(self, stacks):
    self.stack_slugs = [stack['slug'] for stack in stacks]