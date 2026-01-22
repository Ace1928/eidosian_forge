from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import exceptions as api_lib_util_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.projects import util as command_lib_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
def WarnIfSettingProjectWhenAdcExists(project):
    """Warn to update ADC if ADC file contains a different quota_project.

  Args:
    project: a new project to compare with quota_project in the ADC file.

  Returns:
    (Boolean) True if new project does not match the quota_project in the
    ADC file and warning is logged. False otherwise.
  """
    if not os.path.isfile(config.ADCFilePath()):
        return False
    credentials, _ = c_creds.GetGoogleAuthDefault().load_credentials_from_file(config.ADCFilePath())
    if credentials.quota_project_id == project:
        return False
    log.warning('Your active project does not match the quota project in your local Application Default Credentials file. This might result in unexpected quota issues.\n\nTo update your Application Default Credentials quota project, use the `gcloud auth application-default set-quota-project` command.')
    return True