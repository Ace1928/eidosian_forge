from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.utils.listify import listify_lookup_plugin_terms
def _do_flatten(self, terms, variables):
    ret = []
    for term in terms:
        term = self._check_list_of_one_list(term)
        if term == 'None' or term == 'null':
            break
        if isinstance(term, string_types):
            try:
                term2 = listify_lookup_plugin_terms(term, templar=self._templar)
            except TypeError:
                term2 = listify_lookup_plugin_terms(term, templar=self._templar, loader=self._loader)
            if term2 != [term]:
                term = term2
        if isinstance(term, list):
            term = self._do_flatten(term, variables)
            ret.extend(term)
        else:
            ret.append(term)
    return ret