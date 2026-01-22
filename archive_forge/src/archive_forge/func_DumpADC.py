from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from google.auth import jwt
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iamcredentials import util as impersonation_util
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
def DumpADC(credentials, quota_project_disabled=False):
    """Dumps the given credentials to ADC file.

  Args:
     credentials: a credentials from oauth2client or google-auth libraries, the
       credentials to dump.
     quota_project_disabled: bool, If quota project is explicitly disabled by
       users using flags.
  """
    adc_path = c_creds.ADC(credentials).DumpADCToFile()
    LogADCIsWritten(adc_path)
    if quota_project_disabled:
        LogQuotaProjectDisabled()