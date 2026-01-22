from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetRoleName(organization, project, role, attribute='custom roles', parameter_name='ROLE_ID'):
    """Gets the Role name from organization Id and role Id."""
    if role.startswith('roles/'):
        if project or organization:
            raise gcloud_exceptions.InvalidArgumentException(parameter_name, "The role id that starts with 'roles/' only stands for predefined role. Should not specify the project or organization for predefined roles")
        return role
    if role.startswith('projects/') or role.startswith('organizations/'):
        raise gcloud_exceptions.InvalidArgumentException(parameter_name, "The role id should not include any 'projects/' or 'organizations/' prefix.")
    if '/' in role:
        raise gcloud_exceptions.InvalidArgumentException(parameter_name, "The role id should not include any '/' character.")
    VerifyParent(organization, project, attribute)
    if organization:
        return 'organizations/{0}/roles/{1}'.format(organization, role)
    return 'projects/{0}/roles/{1}'.format(project, role)