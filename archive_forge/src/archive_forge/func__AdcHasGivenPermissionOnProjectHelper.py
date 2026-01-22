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
def _AdcHasGivenPermissionOnProjectHelper(project_ref, permissions):
    cred_file_override_old = properties.VALUES.auth.credential_file_override.Get()
    try:
        properties.VALUES.auth.credential_file_override.Set(config.ADCFilePath())
        granted_permissions = projects_api.TestIamPermissions(project_ref, permissions).permissions
        return set(permissions) == set(granted_permissions)
    finally:
        properties.VALUES.auth.credential_file_override.Set(cred_file_override_old)