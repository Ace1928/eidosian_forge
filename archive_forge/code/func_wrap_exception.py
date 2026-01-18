from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
def wrap_exception(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ParseError as e:
        raise AnsibleFilterError(to_text(e))