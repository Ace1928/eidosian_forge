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
def _convert_termination(termination):
    object_app = self._find_app(termination.endpoint.name)
    object_name = ENDPOINT_NAME_MAPPING[termination.endpoint.name]
    return {'object_id': termination.id, 'object_type': f'{object_app}.{object_name}'}