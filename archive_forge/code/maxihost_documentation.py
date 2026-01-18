import re
import json
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.maxihost import MaxihostConnection
from libcloud.common.exceptions import BaseHTTPError

        Create a new SSH key.

        :param      name: Key name (required)
        :type       name: ``str``

        :param      public_key: base64 encoded public key string (required)
        :type       public_key: ``str``
        