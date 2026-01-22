from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class NodeGroupAutoscalingpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'mode': self.request.get('mode'), u'minNodes': self.request.get('min_nodes'), u'maxNodes': self.request.get('max_nodes')})

    def from_response(self):
        return remove_nones_from_dict({u'mode': self.request.get(u'mode'), u'minNodes': self.request.get(u'minNodes'), u'maxNodes': self.request.get(u'maxNodes')})