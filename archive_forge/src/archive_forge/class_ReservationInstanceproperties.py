from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ReservationInstanceproperties(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'machineType': self.request.get('machine_type'), u'minCpuPlatform': self.request.get('min_cpu_platform'), u'guestAccelerators': ReservationGuestacceleratorsArray(self.request.get('guest_accelerators', []), self.module).to_request(), u'localSsds': ReservationLocalssdsArray(self.request.get('local_ssds', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'machineType': self.request.get(u'machineType'), u'minCpuPlatform': self.request.get(u'minCpuPlatform'), u'guestAccelerators': ReservationGuestacceleratorsArray(self.request.get(u'guestAccelerators', []), self.module).from_response(), u'localSsds': ReservationLocalssdsArray(self.request.get(u'localSsds', []), self.module).from_response()})