from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.errors import AnsibleError, AnsibleUndefinedVariable, AnsibleTemplateError
from ansible.module_utils.common.text.converters import to_native
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.display import Display
Loops through the conditionals set on this object, returning
        False if any of them evaluate as such as well as the condition
        that was false.
        