from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _get_model_name_from_url(schema_ref):
    path = schema_ref.split('/')
    return path[len(path) - 1]