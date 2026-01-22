from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
def _ClusterRequest(self, method, api_path, headers=None):
    """Internal method to make requests against the target cluster.

    Args:
      method: request method, e.g. GET
      api_path: path to request against the API server
      headers: dictionary of request headers

    Returns:
      Response body as a string.

    Raises:
      Error: If the response has a status code >= 400.
    """
    if headers is None:
        headers = {}
    self.kube_client.update_params_for_auth(headers=headers, querys=None, auth_settings=['BearerToken'])
    url = urljoin(self.kube_client.configuration.host, api_path)
    r = self.kube_client.rest_client.request(method, url, headers=headers)
    return r.data