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
def CheckApiEnablements(types: List[str]):
    """Checks if all GCP APIs required by the given types are enabled.

  If some required APIs are not enabled, it will prompt the user to enable them.
  If they do not want to enable them, the process will exit.

  Args:
    types: list of types to check.
  """
    project_id = properties.VALUES.core.project.Get()
    apis_not_enabled = []
    for typekit in types:
        try:
            validator = GetIntegrationValidator(typekit)
            apis = validator.GetDisabledGcpApis(project_id)
            if apis:
                apis_not_enabled.extend(apis)
        except ValueError:
            continue
    if apis_not_enabled:
        EnableApis(apis_not_enabled, project_id)