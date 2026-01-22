from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ProjectParent(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'type': self.request.get('type'), u'id': self.request.get('id')})

    def from_response(self):
        return remove_nones_from_dict({u'type': self.request.get(u'type'), u'id': self.request.get(u'id')})