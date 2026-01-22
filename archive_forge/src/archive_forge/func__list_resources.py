import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.compute.types import Provider, NodeState
def _list_resources(self, url, tranform_func):
    data = self.connection.request(url, method='GET').object
    return [tranform_func(obj) for obj in data]