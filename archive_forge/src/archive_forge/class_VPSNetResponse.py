import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
class VPSNetResponse(JsonResponse):

    def parse_body(self):
        try:
            return super().parse_body()
        except MalformedResponseError:
            return self.body

    def success(self):
        if self.status == 406 or self.status == 403:
            raise InvalidCredsError()
        return True

    def parse_error(self):
        try:
            errors = super().parse_body()['errors'][0]
        except MalformedResponseError:
            return self.body
        else:
            return '\n'.join(errors)