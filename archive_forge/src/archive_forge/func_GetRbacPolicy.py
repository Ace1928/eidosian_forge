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
def GetRbacPolicy(self, rbac_to_check):
    """Get the RBAC cluster role binding policy."""
    not_found = False
    for rbac_policy_pair in rbac_to_check:
        rbac_type = rbac_policy_pair[0]
        rbac_name = rbac_policy_pair[1]
        _, err = self._RunKubectl(['get', rbac_type, rbac_name])
        if err:
            if 'NotFound' in err:
                not_found = True
            else:
                raise exceptions.Error('Error retrieving RBAC policy: {}'.format(err))
        else:
            return False
    if not_found:
        return True