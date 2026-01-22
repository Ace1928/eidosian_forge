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
def WriteGcloudCredentialsToADC(creds, add_quota_project=False):
    """Writes gclouds's credential from auth login to ADC json."""
    if not c_creds.IsUserAccountCredentials(creds) and (not c_creds.IsExternalAccountCredentials(creds)):
        log.warning('Credentials cannot be written to application default credentials because it is not a user or external account credential.')
        return
    if c_creds.IsExternalAccountCredentials(creds) and add_quota_project:
        raise AddQuotaProjectError('The application default credentials are external account credentials, quota project cannot be added.')
    PromptIfADCEnvVarIsSet()
    if add_quota_project:
        c_creds.ADC(creds).DumpExtendedADCToFile()
    else:
        c_creds.ADC(creds).DumpADCToFile()