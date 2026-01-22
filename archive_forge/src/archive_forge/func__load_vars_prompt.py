from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.helpers import load_list_of_blocks, load_list_of_roles
from ansible.playbook.role import Role, hash_params
from ansible.playbook.task import Task
from ansible.playbook.taggable import Taggable
from ansible.vars.manager import preprocess_vars
from ansible.utils.display import Display
def _load_vars_prompt(self, attr, ds):
    new_ds = preprocess_vars(ds)
    vars_prompts = []
    if new_ds is not None:
        for prompt_data in new_ds:
            if 'name' not in prompt_data:
                raise AnsibleParserError("Invalid vars_prompt data structure, missing 'name' key", obj=ds)
            for key in prompt_data:
                if key not in ('name', 'prompt', 'default', 'private', 'confirm', 'encrypt', 'salt_size', 'salt', 'unsafe'):
                    raise AnsibleParserError("Invalid vars_prompt data structure, found unsupported key '%s'" % key, obj=ds)
            vars_prompts.append(prompt_data)
    return vars_prompts