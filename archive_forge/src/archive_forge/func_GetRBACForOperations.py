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
def GetRBACForOperations(self, membership, role, project_id, identity, is_user, anthos_support):
    """Get the formatted RBAC policy names."""
    cluster_pattern = re.compile('^clusterrole/')
    namespace_pattern = re.compile('^role/')
    rbac_to_check = []
    if is_user:
        rbac_to_check.extend([('clusterrole', format_util.RbacPolicyName('impersonate', project_id, membership, identity, is_user)), ('clusterrolebinding', format_util.RbacPolicyName('impersonate', project_id, membership, identity, is_user))])
    if anthos_support:
        rbac_to_check.append(('clusterrolebinding', format_util.RbacPolicyName('anthos', project_id, membership, identity, is_user)))
    elif cluster_pattern.match(role.lower()):
        rbac_to_check.append(('clusterrolebinding', format_util.RbacPolicyName('permission', project_id, membership, identity, is_user)))
    elif namespace_pattern.match(role.lower()):
        rbac_to_check.append(('rolebinding', format_util.RbacPolicyName('permission', project_id, membership, identity, is_user)))
    return rbac_to_check