from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class RegionHealthCheckGrpchealthcheck(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'port': self.request.get('port'), u'portName': self.request.get('port_name'), u'portSpecification': self.request.get('port_specification'), u'grpcServiceName': self.request.get('grpc_service_name')})

    def from_response(self):
        return remove_nones_from_dict({u'port': self.request.get(u'port'), u'portName': self.request.get(u'portName'), u'portSpecification': self.request.get(u'portSpecification'), u'grpcServiceName': self.request.get(u'grpcServiceName')})