from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Mapping, Sequence
from jinja2.exceptions import UndefinedError
from ansible.errors import AnsibleLookupError, AnsibleUndefinedVariable
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
def _process_terms(self, terms, variables, kwargs):
    total_search = []
    skip = False
    for term in terms:
        if isinstance(term, Mapping):
            self.set_options(var_options=variables, direct=term)
            files = self.get_option('files')
        elif isinstance(term, string_types):
            files = [term]
        elif isinstance(term, Sequence):
            partial, skip = self._process_terms(term, variables, kwargs)
            total_search.extend(partial)
            continue
        else:
            raise AnsibleLookupError('Invalid term supplied, can handle string, mapping or list of strings but got: %s for %s' % (type(term), term))
        paths = self.get_option('paths')
        skip = self.get_option('skip')
        filelist = _split_on(files, ',;')
        pathlist = _split_on(paths, ',:;')
        if pathlist:
            for path in pathlist:
                for fn in filelist:
                    f = os.path.join(path, fn)
                    total_search.append(f)
        elif filelist:
            total_search.extend(filelist)
        else:
            total_search.append(term)
    return (total_search, skip)