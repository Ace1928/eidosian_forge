from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def EnableApis(apis_not_enabled: List[str], project_id: str):
    """Enables the given API on the given project.

  Args:
    apis_not_enabled: the apis that needs enablement
    project_id: the project ID
  """
    apis_to_enable = '\n\t'.join(apis_not_enabled)
    console_io.PromptContinue(default=False, cancel_on_no=True, message='The following APIs are not enabled on project [{0}]:\n\t{1}'.format(project_id, apis_to_enable), prompt_string=_ConstructPrompt(apis_not_enabled))
    log.status.Print('Enabling APIs on project [{0}]...'.format(project_id))
    op = serviceusage.BatchEnableApiCall(project_id, apis_not_enabled)
    if not op.done:
        op = services_util.WaitOperation(op.name, serviceusage.GetOperation)
        services_util.PrintOperation(op)