from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerSource(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'storageSource': TriggerStoragesource(self.request.get('storage_source', {}), self.module).to_request(), u'repoSource': TriggerReposource(self.request.get('repo_source', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'storageSource': TriggerStoragesource(self.request.get(u'storageSource', {}), self.module).from_response(), u'repoSource': TriggerReposource(self.request.get(u'repoSource', {}), self.module).from_response()})