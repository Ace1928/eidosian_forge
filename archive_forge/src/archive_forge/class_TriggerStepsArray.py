from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerStepsArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({u'name': item.get('name'), u'args': item.get('args'), u'env': item.get('env'), u'id': item.get('id'), u'entrypoint': item.get('entrypoint'), u'dir': item.get('dir'), u'secretEnv': item.get('secret_env'), u'timeout': item.get('timeout'), u'timing': item.get('timing'), u'volumes': TriggerVolumesArray(item.get('volumes', []), self.module).to_request(), u'waitFor': item.get('wait_for')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'name': item.get(u'name'), u'args': item.get(u'args'), u'env': item.get(u'env'), u'id': item.get(u'id'), u'entrypoint': item.get(u'entrypoint'), u'dir': item.get(u'dir'), u'secretEnv': item.get(u'secretEnv'), u'timeout': item.get(u'timeout'), u'timing': item.get(u'timing'), u'volumes': TriggerVolumesArray(item.get(u'volumes', []), self.module).from_response(), u'waitFor': item.get(u'waitFor')})