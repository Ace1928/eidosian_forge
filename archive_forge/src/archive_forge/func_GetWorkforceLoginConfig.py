from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account_authorized_user
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
def GetWorkforceLoginConfig():
    """_GetWorkforceLoginConfig gets the correct Credential Configuration.

  It will first check from the supplied argument if present, then from an
  environment variable if present, and finally from the project settings, if
  present.

  Returns:
    Optional[str]: The name of the Credential Configuration File to use.
  """
    return properties.VALUES.auth.login_config_file.Get()