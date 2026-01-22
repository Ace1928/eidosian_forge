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
def ExtractAndValidateAccount(account, creds):
    """Extracts account from creds and validates it against account."""
    decoded_id_token = jwt.decode(creds.id_token, verify=False)
    web_flow_account = decoded_id_token['email']
    if account and account.lower() != web_flow_account.lower():
        raise auth_exceptions.WrongAccountError('You attempted to log in as account [{account}] but the received credentials were for account [{web_flow_account}].\n\nPlease check that your browser is logged in as account [{account}] and that you are using the correct browser profile.'.format(account=account, web_flow_account=web_flow_account))
    return web_flow_account