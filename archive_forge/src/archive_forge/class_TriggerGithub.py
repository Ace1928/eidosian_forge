from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerGithub(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'owner': self.request.get('owner'), u'name': self.request.get('name'), u'pullRequest': TriggerPullrequest(self.request.get('pull_request', {}), self.module).to_request(), u'push': TriggerPush(self.request.get('push', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'owner': self.request.get(u'owner'), u'name': self.request.get(u'name'), u'pullRequest': TriggerPullrequest(self.request.get(u'pullRequest', {}), self.module).from_response(), u'push': TriggerPush(self.request.get(u'push', {}), self.module).from_response()})