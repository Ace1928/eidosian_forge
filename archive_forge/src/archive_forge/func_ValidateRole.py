from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet import invalid_args_error
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.command_lib.container.fleet.memberships import errors as memberships_errors
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
def ValidateRole(role):
    """Validation for the role in correct format."""
    cluster_pattern = re.compile('^clusterrole/')
    namespace_pattern = re.compile('^role/')
    if cluster_pattern.match(role.lower()):
        log.status.Print('Specified Cluster Role is:', role)
        if len(role.split('/')) != 2:
            raise invalid_args_error.InvalidArgsError('Cluster role is not specified in correct format. Please specify the cluster role as: clusterrole/cluster-permission.')
    elif namespace_pattern.match(role.lower()):
        log.status.Print('Specified Namespace Role is:', role)
        if len(role.split('/')) != 3:
            raise invalid_args_error.InvalidArgsError('Namespace role is not specified in correct format. Please specify the namespace role as: role/namespace/namespace-permission')
    else:
        raise invalid_args_error.InvalidArgsError('The required role is not a cluster role or a namespace role.')