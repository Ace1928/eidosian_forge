from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.sentinel import Sentinel
def _flatten_tags(tags: list) -> list:
    rv = set()
    for tag in tags:
        if isinstance(tag, list):
            rv.update(tag)
        else:
            rv.add(tag)
    return list(rv)