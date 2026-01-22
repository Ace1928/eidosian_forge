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
def _update_netbox_object(self, data):
    """Update a NetBox object.
        :returns tuple(serialized_nb_obj, diff): tuple of the serialized updated
        NetBox object and the Ansible diff.
        """
    serialized_nb_obj = self.nb_object.serialize()
    if 'custom_fields' in serialized_nb_obj:
        custom_fields = serialized_nb_obj.get('custom_fields', {})
        shared_keys = custom_fields.keys() & data.get('custom_fields', {}).keys()
        serialized_nb_obj['custom_fields'] = {key: custom_fields[key] for key in shared_keys if custom_fields[key] is not None}
    updated_obj = serialized_nb_obj.copy()
    updated_obj.update(data)
    if serialized_nb_obj.get('tags') and data.get('tags'):
        serialized_nb_obj['tags'] = set(serialized_nb_obj['tags'])
        updated_obj['tags'] = set(data['tags'])
    version_pre_30 = self._version_check_greater('3.0', self.version)
    if serialized_nb_obj.get('latitude') and data.get('latitude') and version_pre_30:
        updated_obj['latitude'] = str(data['latitude'])
    if serialized_nb_obj.get('longitude') and data.get('longitude') and version_pre_30:
        updated_obj['longitude'] = str(data['longitude'])
    version_pre_211 = self._version_check_greater('2.11', self.version)
    if serialized_nb_obj.get('vcpus') and data.get('vcpus'):
        if version_pre_211:
            updated_obj['vcpus'] = int(data['vcpus'])
        else:
            updated_obj['vcpus'] = float(data['vcpus'])
    version_post_33 = self._version_check_greater(self.version, '3.3', True)
    if serialized_nb_obj.get('a_terminations') and serialized_nb_obj.get('b_terminations') and data.get('a_terminations') and data.get('b_terminations') and version_post_33:

        def _convert_termination(termination):
            object_app = self._find_app(termination.endpoint.name)
            object_name = ENDPOINT_NAME_MAPPING[termination.endpoint.name]
            return {'object_id': termination.id, 'object_type': f'{object_app}.{object_name}'}
        serialized_nb_obj['a_terminations'] = list(map(_convert_termination, self.nb_object.a_terminations))
        serialized_nb_obj['b_terminations'] = list(map(_convert_termination, self.nb_object.b_terminations))
    if serialized_nb_obj == updated_obj:
        return (serialized_nb_obj, None)
    else:
        data_before, data_after = ({}, {})
        for key in data:
            try:
                if serialized_nb_obj[key] != updated_obj[key]:
                    data_before[key] = serialized_nb_obj[key]
                    data_after[key] = updated_obj[key]
            except KeyError:
                if key == 'form_factor':
                    msg = 'form_factor is not valid for NetBox 2.7 onward. Please use the type key instead.'
                else:
                    msg = '%s does not exist on existing object. Check to make sure valid field.' % key
                self._handle_errors(msg=msg)
        if not self.check_mode:
            self.nb_object.update(data)
            updated_obj = self.nb_object.serialize()
        diff = self._build_diff(before=data_before, after=data_after)
        return (updated_obj, diff)