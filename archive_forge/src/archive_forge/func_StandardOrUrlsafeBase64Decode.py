import base64
import binascii
import collections
import copy
import json
from typing import List, Optional, Text
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_session
from containerregistry.transform.v2_2 import metadata
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_util
from googlecloudsdk.core.exceptions import Error
import httplib2
def StandardOrUrlsafeBase64Decode(encoded):
    try:
        return base64.b64decode(encoded)
    except binascii.Error:
        return base64.urlsafe_b64decode(AddMissingBase64Padding(encoded))