from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceScheduling(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'automaticRestart': self.request.get('automatic_restart'), u'onHostMaintenance': self.request.get('on_host_maintenance'), u'preemptible': self.request.get('preemptible')})

    def from_response(self):
        return remove_nones_from_dict({u'automaticRestart': self.request.get(u'automaticRestart'), u'onHostMaintenance': self.request.get(u'onHostMaintenance'), u'preemptible': self.request.get(u'preemptible')})