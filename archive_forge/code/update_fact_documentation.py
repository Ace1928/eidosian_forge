from __future__ import absolute_import, division, print_function
import ast
import re
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.action import ActionBase
from jinja2 import Template, TemplateSyntaxError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.update_fact import DOCUMENTATION
action entry point