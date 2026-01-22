from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_by_ref(self, model_prop_val):
    model = _get_model_name_from_url(model_prop_val[PropName.REF])
    return self._models[model]