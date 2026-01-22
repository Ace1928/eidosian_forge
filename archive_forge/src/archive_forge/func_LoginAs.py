from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.auth import external_account as auth_external_account
from googlecloudsdk.api_lib.auth import service_account as auth_service_account
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.auth import auth_util as command_auth_util
from googlecloudsdk.command_lib.auth import flags as auth_flags
from googlecloudsdk.command_lib.auth import workforce_login_config as workforce_login_config_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.credentials import store as c_store
def LoginAs(account, creds, project, activate, brief, update_adc, add_quota_project_to_adc):
    """Logs in with valid credentials."""
    if hasattr(creds, 'universe_domain'):
        auth_util.HandleUniverseDomainConflict(creds.universe_domain, account)
    _ValidateADCFlags(update_adc, add_quota_project_to_adc)
    if update_adc:
        _UpdateADC(creds, add_quota_project_to_adc)
    if not activate:
        return creds
    properties.PersistProperty(properties.VALUES.core.account, account)
    if project:
        properties.PersistProperty(properties.VALUES.core.project, project)
    if not brief:
        if c_creds.IsExternalAccountCredentials(creds):
            confirmation_msg = 'Authenticated with external account credentials for: [{0}].'.format(account)
        elif c_creds.IsExternalAccountUserCredentials(creds):
            confirmation_msg = 'Authenticated with external account user credentials for: [{0}].'.format(account)
        elif c_creds.IsServiceAccountCredentials(creds):
            confirmation_msg = 'Authenticated with service account credentials for: [{0}].'.format(account)
        elif c_creds.IsExternalAccountAuthorizedUserCredentials(creds):
            confirmation_msg = 'Authenticated with external account authorized user credentials for: [{0}].'.format(account)
        else:
            confirmation_msg = 'You are now logged in as [{0}].'.format(account)
        log.status.write('\n{confirmation_msg}\nYour current project is [{project}].  You can change this setting by running:\n  $ gcloud config set project PROJECT_ID\n'.format(confirmation_msg=confirmation_msg, project=properties.VALUES.core.project.Get()))
    return creds