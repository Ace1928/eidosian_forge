import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class VSphereConnection(ConnectionKey):
    responseCls = VSphereResponse
    session_token = None

    def add_default_headers(self, headers):
        """
        VSphere needs an initial connection to a specific API endpoint to
        generate a session-token, which will be used for the purpose of
        authenticating for the rest of the session.
        """
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'
        if self.session_token is None:
            to_encode = '{}:{}'.format(self.key, self.secret)
            b64_user_pass = base64.b64encode(to_encode.encode())
            headers['Authorization'] = 'Basic {}'.format(b64_user_pass.decode())
        else:
            headers['vmware-api-session-id'] = self.session_token
        return headers