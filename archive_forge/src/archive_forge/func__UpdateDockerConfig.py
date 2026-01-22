from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def _UpdateDockerConfig(server, username, access_token):
    """Register the username / token for the given server on Docker's keyring."""
    try:
        dockercfg_contents = ReadDockerAuthConfig()
    except (IOError, client_lib.InvalidDockerConfigError):
        dockercfg_contents = {}
    auth = username + ':' + access_token
    auth = base64.b64encode(auth.encode('ascii')).decode('ascii')
    parsed_url = client_lib.GetNormalizedURL(server)
    server = parsed_url.geturl()
    server_unqualified = parsed_url.hostname
    if server_unqualified in dockercfg_contents:
        del dockercfg_contents[server_unqualified]
    dockercfg_contents[server] = {'auth': auth, 'email': _EMAIL}
    WriteDockerAuthConfig(dockercfg_contents)