from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
def AllAccountsWithUniverseDomains():
    """Get all accounts and universe domains for the auth command Run() method.

  Returns:
    List[AccInfoWithUniverseDomain]: The list of account and universe domain
      information for all accounts.
  """
    store = c_creds.GetCredentialStore(with_access_token_cache=False)
    accounts_dict = store.GetAccountsWithUniverseDomain()
    static_accounts = STATIC_CREDENTIAL_PROVIDERS.GetAccounts()
    for account in static_accounts:
        if account not in accounts_dict:
            creds = STATIC_CREDENTIAL_PROVIDERS.GetCredentials(account)
            accounts_dict[account] = [creds.universe_domain if hasattr(creds, 'universe_domain') else properties.VALUES.core.universe_domain.default]
    accounts_dict = dict(sorted(accounts_dict.items()))
    active_account = properties.VALUES.core.account.Get()
    universe_domain_property = properties.VALUES.core.universe_domain.Get()
    result = []
    for account in accounts_dict:
        for universe_domain in accounts_dict[account]:
            is_active = account == active_account and universe_domain_property == universe_domain
            result.append(AcctInfoWithUniverseDomain(account, is_active, universe_domain))
    return result