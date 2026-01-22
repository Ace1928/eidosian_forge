from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _find_ids(self, data, user_query_params):
    """Will find the IDs of all user specified data if resolvable
        :returns data (dict): Returns the updated dict with the IDs of user specified data
        :params data (dict): User defined data passed into the module
        """
    for k, v in data.items():
        if k in CONVERT_TO_ID:
            if not self._version_check_greater(self.version, '2.9', greater_or_equal=True) and k == 'tags' or (self.endpoint == 'config_contexts' and k == 'tags'):
                continue
            if k == 'termination_a':
                endpoint = CONVERT_TO_ID[data.get('termination_a_type')]
            elif k == 'termination_b':
                endpoint = CONVERT_TO_ID[data.get('termination_b_type')]
            elif k == 'assigned_object':
                endpoint = 'interfaces'
            elif k == 'component':
                endpoint = CONVERT_TO_ID[data.get('component_type')]
            elif k == 'scope':
                endpoint = SCOPE_TO_ENDPOINT[data['scope_type']]
            else:
                endpoint = CONVERT_TO_ID[k]
            search = v
            app = self._find_app(endpoint)
            nb_app = getattr(self.nb, app)
            nb_endpoint = getattr(nb_app, endpoint)
            if isinstance(v, dict):
                if (k == 'interface' or k == 'assigned_object') and v.get('virtual_machine'):
                    nb_app = getattr(self.nb, 'virtualization')
                    nb_endpoint = getattr(nb_app, endpoint)
                query_params = self._build_query_params(k, data, child=v)
                query_id = self._nb_endpoint_get(nb_endpoint, query_params, k)
            elif isinstance(v, list):
                id_list = list()
                for list_item in v:
                    if k in ('regions', 'sites', 'roles', 'device_types', 'platforms', 'cluster_groups', 'contact_groups', 'tenant_groups', 'tenants', 'tags') and isinstance(list_item, str):
                        temp_dict = {'slug': self._to_slug(list_item)}
                    elif isinstance(list_item, dict):
                        norm_data = self._normalize_data(list_item)
                        temp_dict = self._build_query_params(k, data, child=norm_data)
                    elif isinstance(list_item, int):
                        id_list.append(list_item)
                        continue
                    else:
                        temp_dict = {QUERY_TYPES.get(k, 'q'): list_item}
                    query_id = self._nb_endpoint_get(nb_endpoint, temp_dict, k)
                    if query_id:
                        id_list.append(query_id.id)
                    else:
                        self._handle_errors(msg='%s not found' % list_item)
            else:
                if k in ['lag', 'parent_interface', 'rear_port', 'rear_port_template', 'power_port', 'power_port_template']:
                    query_params = self._build_query_params(k, data, user_query_params)
                elif k == 'scope':
                    query_params = {QUERY_TYPES.get(ENDPOINT_NAME_MAPPING[endpoint], 'q'): search}
                elif k == 'parent_vm_interface':
                    nb_app = getattr(self.nb, 'virtualization')
                    nb_endpoint = getattr(nb_app, endpoint)
                    query_params = self._build_query_params(k, data, user_query_params)
                elif k == 'vm_bridge':
                    nb_app = getattr(self.nb, 'virtualization')
                    nb_endpoint = getattr(nb_app, endpoint)
                    query_params = self._build_query_params(k, data, user_query_params)
                else:
                    query_params = {QUERY_TYPES.get(k, 'q'): search}
                query_id = self._nb_endpoint_get(nb_endpoint, query_params, k)
            if isinstance(v, list):
                data[k] = id_list
            elif isinstance(v, int):
                pass
            elif query_id:
                data[k] = query_id.id
            else:
                self._handle_errors(msg='Could not resolve id of %s: %s' % (k, v))
    return data