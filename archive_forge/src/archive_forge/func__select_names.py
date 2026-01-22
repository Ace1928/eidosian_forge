from __future__ import absolute_import, division, print_function
import traceback
from contextlib import contextmanager
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def _select_names(spec):
    dep_names = sorted(_deps)
    if spec:
        if spec.startswith('-'):
            spec_split = spec[1:].split(':')
            for d in spec_split:
                dep_names.remove(d)
        else:
            spec_split = spec.split(':')
            dep_names = []
            for d in spec_split:
                _deps[d]
                dep_names.append(d)
    return dep_names