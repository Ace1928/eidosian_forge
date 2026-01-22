from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
def GetServices():
    response = requests.GetSession().get(_SERVICE_CATALOG_URL)
    catalog = json.loads(response.text)
    return catalog['services']