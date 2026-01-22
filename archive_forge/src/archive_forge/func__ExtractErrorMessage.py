from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import requests
from six.moves import urllib
def _ExtractErrorMessage(response):
    """Extracts error message from response, returns None if message not found."""
    json_response = json.loads(response)
    if ERROR_FIELD in json_response and isinstance(json_response[ERROR_FIELD], dict) and (MESSAGE_FIELD in json_response[ERROR_FIELD]):
        return json_response[ERROR_FIELD][MESSAGE_FIELD]
    return None