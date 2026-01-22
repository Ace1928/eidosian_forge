from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.sentinel import Sentinel
def _load_tags(self, attr, ds):
    if isinstance(ds, list):
        return ds
    elif isinstance(ds, string_types):
        return [x.strip() for x in ds.split(',')]
    else:
        raise AnsibleError('tags must be specified as a list', obj=ds)