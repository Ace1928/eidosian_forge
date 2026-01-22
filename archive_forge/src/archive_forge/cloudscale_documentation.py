import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.compute.types import Provider, NodeState

        Delete a node. It's also possible to use ``node.destroy()``.
        This will irreversibly delete the cloudscale.ch server and all its
        volumes. So please be cautious.
        