from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _ParseDomainPath(self, url):
    url_parts = url.split('/', 1)
    domain = url_parts[0]
    path = '/*'
    if len(url_parts) == 2:
        path = '/' + url_parts[1]
    return (domain.lower(), path)