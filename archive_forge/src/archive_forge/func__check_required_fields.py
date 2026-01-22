from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _check_required_fields(self, status, required_fields, data, path):
    missed_required_fields = [self._create_path_to_field(path, field) for field in required_fields if field not in data.keys() or data[field] is None]
    if len(missed_required_fields) > 0:
        status[PropName.REQUIRED] += missed_required_fields