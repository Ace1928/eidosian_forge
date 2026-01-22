from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import requests
from six.moves import http_client as httplib
def _GetPrediction(url, body, headers):
    """Make http request to get prediction results."""
    response = requests.GetSession().request('POST', url, data=body, headers=headers)
    return (response.status_code, response.text)