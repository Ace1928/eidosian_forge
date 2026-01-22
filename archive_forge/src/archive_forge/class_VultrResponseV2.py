from typing import Any, Dict, Optional
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import VolumeSnapshot
class VultrResponseV2(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.CREATED, httplib.ACCEPTED, httplib.NO_CONTENT]

    def parse_error(self):
        """
        Parse the error body and raise the appropriate exception
        """
        status = self.status
        data = self.parse_body()
        error_msg = data.get('error', '')
        raise VultrException(code=status, message=error_msg)

    def success(self):
        """Check the response for success

        :return: ``bool`` indicating a successful request
        """
        return self.status in self.valid_response_codes