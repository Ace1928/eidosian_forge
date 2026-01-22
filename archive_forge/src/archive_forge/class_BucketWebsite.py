from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class BucketWebsite(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'mainPageSuffix': self.request.get('main_page_suffix'), u'notFoundPage': self.request.get('not_found_page')})

    def from_response(self):
        return remove_nones_from_dict({u'mainPageSuffix': self.request.get(u'mainPageSuffix'), u'notFoundPage': self.request.get(u'notFoundPage')})