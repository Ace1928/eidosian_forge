import json
import time
import datetime
from libcloud.utils.py3 import basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
class KamateraResponse(JsonResponse):
    """
    Response class for KamateraDriver
    """

    def parse_error(self):
        data = self.parse_body()
        if 'message' in data:
            return data['message']
        else:
            return json.dumps(data)