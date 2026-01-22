from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def HasNamespaceReaderPermissions(self, *namespaces):
    """Check to see if the user has read permissions in the namespaces.

    Args:
      *namespaces: The namespaces to verify reader permissions.

    Returns:
      true, if reads can be performed on all of the specified namespaces.

    Raises:
      Error: if failing to get check for read permissions.
      Error: if read permissions are not found.
    """
    for ns in namespaces:
        out, err = self._RunKubectl(['auth', 'can-i', 'get', '*', '-n', ns], None)
        if err:
            raise exceptions.Error('Failed to check if the user can read resources in {} namespace: {}'.format(ns, err))
        if 'yes' not in out:
            raise exceptions.Error('Missing permissions to read resources in {} namespace'.format(ns))
    return True