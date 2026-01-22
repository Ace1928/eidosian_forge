from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapQueryparametermatchesArray(object):

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
        return remove_nones_from_dict({u'exactMatch': item.get('exact_match'), u'name': item.get('name'), u'presentMatch': item.get('present_match'), u'regexMatch': item.get('regex_match')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'exactMatch': item.get(u'exactMatch'), u'name': item.get(u'name'), u'presentMatch': item.get(u'presentMatch'), u'regexMatch': item.get(u'regexMatch')})