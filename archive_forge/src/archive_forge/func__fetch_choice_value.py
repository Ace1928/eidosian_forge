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
def _fetch_choice_value(self, search, endpoint):
    app = self._find_app(endpoint)
    nb_app = getattr(self.nb, app)
    nb_endpoint = getattr(nb_app, endpoint)
    try:
        endpoint_choices = nb_endpoint.choices()
    except ValueError:
        self._handle_errors(msg='Failed to fetch endpoint choices to validate against. This requires a write-enabled token. Make sure the token is write-enabled. If looking to fetch only information, use either the inventory or lookup plugin.')
    choices = list(chain.from_iterable(endpoint_choices.values()))
    for item in choices:
        if item['display_name'].lower() == search.lower():
            return item['value']
        elif item['value'] == search.lower():
            return item['value']
    self._handle_errors(msg='%s was not found as a valid choice for %s' % (search, endpoint))