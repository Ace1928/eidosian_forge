from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicySnapshotschedulepolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'schedule': ResourcePolicySchedule(self.request.get('schedule', {}), self.module).to_request(), u'retentionPolicy': ResourcePolicyRetentionpolicy(self.request.get('retention_policy', {}), self.module).to_request(), u'snapshotProperties': ResourcePolicySnapshotproperties(self.request.get('snapshot_properties', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'schedule': ResourcePolicySchedule(self.request.get(u'schedule', {}), self.module).from_response(), u'retentionPolicy': ResourcePolicyRetentionpolicy(self.request.get(u'retentionPolicy', {}), self.module).from_response(), u'snapshotProperties': ResourcePolicySnapshotproperties(self.request.get(u'snapshotProperties', {}), self.module).from_response()})