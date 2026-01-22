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
def _get_query_param_id(self, match, data):
    """Used to find IDs of necessary searches when required under _build_query_params
        :returns id (int) or data (dict): Either returns the ID or original data passed in
        :params match (str): The key within the user defined data that is required to have an ID
        :params data (dict): User defined data passed into the module
        """
    if isinstance(data.get(match), int):
        return data[match]
    endpoint = CONVERT_TO_ID[match]
    app = self._find_app(endpoint)
    nb_app = getattr(self.nb, app)
    nb_endpoint = getattr(nb_app, endpoint)
    query_params = {QUERY_TYPES.get(match): data[match]}
    result = self._nb_endpoint_get(nb_endpoint, query_params, match)
    if result:
        return result.id
    else:
        return data