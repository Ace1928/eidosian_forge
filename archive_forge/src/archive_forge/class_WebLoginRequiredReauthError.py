from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class WebLoginRequiredReauthError(Error):
    """An exception raised when login through browser is required for reauth.

  This applies to SAML users who set password as their reauth method today.
  Since SAML uers do not have knowledge of their Google password, we require
  web login and allow users to be authenticated by their IDP.
  """

    def __init__(self, for_adc=False):
        login_command = ADC_LOGIN_COMMAND if for_adc else AUTH_LOGIN_COMMAND
        super(WebLoginRequiredReauthError, self).__init__(textwrap.dedent('        Please run:\n\n          $ {login_command}\n\n        to complete reauthentication.'.format(login_command=login_command)))