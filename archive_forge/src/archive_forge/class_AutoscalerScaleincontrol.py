from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class AutoscalerScaleincontrol(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'maxScaledInReplicas': AutoscalerMaxscaledinreplicas(self.request.get('max_scaled_in_replicas', {}), self.module).to_request(), u'timeWindowSec': self.request.get('time_window_sec')})

    def from_response(self):
        return remove_nones_from_dict({u'maxScaledInReplicas': AutoscalerMaxscaledinreplicas(self.request.get(u'maxScaledInReplicas', {}), self.module).from_response(), u'timeWindowSec': self.request.get(u'timeWindowSec')})