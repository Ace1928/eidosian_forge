from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_name_from_delete_operation(self, params):
    operation_id = params[PropName.OPERATION_ID]
    if operation_id.startswith(DELETE_PREFIX):
        model_name = operation_id[len(DELETE_PREFIX):]
        if model_name in self._definitions:
            return model_name
    return None