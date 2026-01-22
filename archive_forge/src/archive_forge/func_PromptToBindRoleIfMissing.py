from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util as projects_api_util
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding as encoder
from googlecloudsdk.core.util import retry
import six
def PromptToBindRoleIfMissing(sa_email, role, alt_roles=None, reason=''):
    """Prompts to bind the role to the service account if missing.

  If the console cannot prompt, a warning is logged instead.

  Args:
    sa_email: The service account email to bind the role to.
    role: The role to bind if missing.
    alt_roles: Alternative roles to check that dismiss the need to bind the
      specified role.
    reason: Extra information to print explaining why the binding is necessary.
  """
    alt_roles = alt_roles or []
    project_ref = projects_util.ParseProject(GetProject())
    member = 'serviceAccount:{}'.format(sa_email)
    try:
        iam_policy = projects_api.GetIamPolicy(project_ref)
        if any((HasRoleBinding(iam_policy, sa_email, r) for r in [role, *alt_roles])):
            return
        log.status.Print('Service account [{}] is missing the role [{}].\n{}'.format(sa_email, role, reason))
        bind = console_io.CanPrompt() and console_io.PromptContinue(prompt_string='\nBind the role [{}] to service account [{}]?'.format(role, sa_email))
        if not bind:
            log.warning('Manual binding of above role may be necessary.\n')
            return
        projects_api.AddIamPolicyBinding(project_ref, member, role)
        log.status.Print('Role successfully bound.\n')
    except apitools_exceptions.HttpForbiddenError:
        log.warning('Your account does not have permission to check or bind IAM policies to project [%s]. If the deployment fails, ensure [%s] has the role [%s] before retrying.', project_ref, sa_email, role)