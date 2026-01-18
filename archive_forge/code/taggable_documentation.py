from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.sentinel import Sentinel
 this checks if the current item should be executed depending on tag options 