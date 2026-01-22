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
def LoginWithCredFileConfig(cred_config, scopes, project, activate, brief, update_adc, add_quota_project_to_adc, args_account):
    """Login with the provided configuration loaded from --cred-file.

  Args:
    cred_config (Mapping): The configuration dictionary representing the
      credentials. This is loaded from the --cred-file argument.
    scopes (Tuple[str]): The default OAuth scopes to use.
    project (Optional[str]): The optional project ID to activate / persist.
    activate (bool): Whether to set the new account associated with the
      credentials to active.
    brief (bool): Whether to use minimal user output.
    update_adc (bool): Whether to write the obtained credentials to the
      well-known location for Application Default Credentials (ADC).
    add_quota_project_to_adc (bool): Whether to add the quota project to the
      application default credentials file.
    args_account (Optional[str]): The optional ACCOUNT argument. When provided,
      this should match the account ID on the authenticated credentials.

  Returns:
    google.auth.credentials.Credentials: The authenticated stored credentials.

  Raises:
    calliope_exceptions.ConflictingArgumentsException: If conflicting arguments
      are provided.
    calliope_exceptions.InvalidArgumentException: If invalid arguments are
      provided.
  """
    scopes = tuple((x for x in scopes if x != config.REAUTH_SCOPE))
    if add_quota_project_to_adc:
        raise calliope_exceptions.ConflictingArgumentsException('[--add-quota-project-to-adc] cannot be specified with --cred-file')
    if auth_external_account.IsExternalAccountConfig(cred_config):
        creds = auth_external_account.CredentialsFromAdcDictGoogleAuth(cred_config)
        account = auth_external_account.GetExternalAccountId(creds)
        if hasattr(creds, 'interactive') and creds.interactive:
            setattr(creds, '_tokeninfo_username', account)
    elif auth_service_account.IsServiceAccountConfig(cred_config):
        creds = auth_service_account.CredentialsFromAdcDictGoogleAuth(cred_config)
        account = creds.service_account_email
    else:
        raise calliope_exceptions.InvalidArgumentException('--cred-file', 'Only external account or service account JSON credential file types are supported.')
    if args_account and args_account != account:
        raise calliope_exceptions.InvalidArgumentException('ACCOUNT', 'The given account name does not match the account name in the credential file. This argument can be omitted when using credential files.')
    try:
        exist_creds = c_store.Load(account=account, scopes=scopes, prevent_refresh=True)
    except creds_exceptions.Error:
        exist_creds = None
    if exist_creds and exist_creds.universe_domain == creds.universe_domain:
        message = textwrap.dedent("\n      You are already authenticated with '%s'.\n      Do you wish to proceed and overwrite existing credentials?\n      ")
        answer = console_io.PromptContinue(message=message % account, default=True)
        if not answer:
            return None
    c_store.Store(creds, account, scopes=scopes)
    return LoginAs(account, creds, project, activate, brief, update_adc, False)