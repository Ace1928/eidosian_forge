import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
class UpcloudResponse(JsonResponse):
    """
    Response class for UpcloudDriver
    """

    def success(self):
        if self.status == httplib.NO_CONTENT:
            return True
        return super().success()

    def parse_error(self):
        data = self.parse_body()
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(value=data['error']['error_message'])
        return data