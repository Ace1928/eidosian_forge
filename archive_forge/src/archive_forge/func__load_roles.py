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
def _load_roles(self, attr, ds):
    """
        Loads and returns a list of RoleInclude objects from the datastructure
        list of role definitions and creates the Role from those objects
        """
    if ds is None:
        ds = []
    try:
        role_includes = load_list_of_roles(ds, play=self, variable_manager=self._variable_manager, loader=self._loader, collection_search_list=self.collections)
    except AssertionError as e:
        raise AnsibleParserError('A malformed role declaration was encountered.', obj=self._ds, orig_exc=e)
    roles = []
    for ri in role_includes:
        roles.append(Role.load(ri, play=self))
    self.roles[:0] = roles
    return self.roles