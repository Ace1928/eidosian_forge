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
def _ensure_object_absent(self, endpoint_name, name):
    """Used when `state` is absent to make sure object does not exist
        :params endpoint_name (str): Endpoint name that was created/updated. ex. device
        :params name (str): Name of the object
        """
    if self.nb_object:
        diff = self._delete_netbox_object()
        self.result['msg'] = '%s %s deleted' % (endpoint_name, name)
        self.result['changed'] = True
        self.result['diff'] = diff
    else:
        self.result['msg'] = '%s %s already absent' % (endpoint_name, name)