from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapHeadermatchesArray(object):

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
        return remove_nones_from_dict({u'exactMatch': item.get('exact_match'), u'headerName': item.get('header_name'), u'invertMatch': item.get('invert_match'), u'prefixMatch': item.get('prefix_match'), u'presentMatch': item.get('present_match'), u'rangeMatch': UrlMapRangematch(item.get('range_match', {}), self.module).to_request(), u'regexMatch': item.get('regex_match'), u'suffixMatch': item.get('suffix_match')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'exactMatch': item.get(u'exactMatch'), u'headerName': item.get(u'headerName'), u'invertMatch': item.get(u'invertMatch'), u'prefixMatch': item.get(u'prefixMatch'), u'presentMatch': item.get(u'presentMatch'), u'rangeMatch': UrlMapRangematch(item.get(u'rangeMatch', {}), self.module).from_response(), u'regexMatch': item.get(u'regexMatch'), u'suffixMatch': item.get(u'suffixMatch')})