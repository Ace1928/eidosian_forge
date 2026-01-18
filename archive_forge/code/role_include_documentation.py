from __future__ import (absolute_import, division, print_function)
from os.path import basename
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task_include import TaskInclude
from ansible.playbook.role import Role
from ansible.playbook.role.include import RoleInclude
from ansible.utils.display import Display
from ansible.module_utils.six import string_types
from ansible.template import Templar
 return the name of the task 