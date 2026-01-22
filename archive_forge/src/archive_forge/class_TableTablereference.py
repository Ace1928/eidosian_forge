from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableTablereference(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'datasetId': self.request.get('dataset_id'), u'projectId': self.request.get('project_id'), u'tableId': self.request.get('table_id')})

    def from_response(self):
        return remove_nones_from_dict({u'datasetId': self.request.get(u'datasetId'), u'projectId': self.request.get(u'projectId'), u'tableId': self.request.get(u'tableId')})