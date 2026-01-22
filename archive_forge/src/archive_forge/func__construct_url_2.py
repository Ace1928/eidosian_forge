from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def _construct_url_2(self, parent, obj, config_only=True):
    """
        This method is used by construct_url when the object is the second-level class.
        """
    parent_rn = parent.get('aci_rn')
    parent_obj = parent.get('module_object')
    obj_class = obj.get('aci_class')
    obj_rn = obj.get('aci_rn')
    obj_filter = obj.get('target_filter')
    mo = obj.get('module_object')
    if self.module.params.get('state') in ('absent', 'present'):
        self.path = 'api/mo/uni/{0}/{1}.json'.format(parent_rn, obj_rn)
        self.parent_path = 'api/mo/uni/{0}.json'.format(parent_rn)
        if config_only:
            self.update_qs({'rsp-prop-include': 'config-only'})
        self.obj_filter = obj_filter
    elif parent_obj is None and mo is None:
        self.path = 'api/class/{0}.json'.format(obj_class)
        self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
    elif parent_obj is None:
        self.path = 'api/class/{0}.json'.format(obj_class)
        self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
    elif mo is None:
        self.child_classes.add(obj_class)
        self.path = 'api/mo/uni/{0}.json'.format(parent_rn)
    else:
        self.path = 'api/mo/uni/{0}/{1}.json'.format(parent_rn, obj_rn)