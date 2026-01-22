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
class GceCredentialProvider(object):
    """Provides account, project and credential data for gce vm env."""

    def GetCredentials(self, account, use_google_auth=True):
        if account in c_gce.Metadata().Accounts():
            refresh = not use_google_auth
            return AcquireFromGCE(account, use_google_auth, refresh)
        return None

    def GetAccount(self):
        if properties.VALUES.core.check_gce_metadata.GetBool():
            return c_gce.Metadata().DefaultAccount()
        return None

    def GetAccounts(self):
        return set(c_gce.Metadata().Accounts())

    def GetUniverseDomain(self):
        """Gets the universe domain from GCE metadata.

    Returns:
      str: The universe domain from metadata server. Returns None if
        core/check_gce_metadata property is False.
    """
        if properties.VALUES.core.check_gce_metadata.GetBool():
            return c_gce.Metadata().UniverseDomain()
        return None

    def GetProject(self):
        if properties.VALUES.core.check_gce_metadata.GetBool():
            return c_gce.Metadata().Project()
        return None

    def Register(self):
        properties.VALUES.core.account.AddCallback(self.GetAccount)
        properties.VALUES.core.project.AddCallback(self.GetProject)
        properties.VALUES.core.universe_domain.AddCallback(self.GetUniverseDomain)
        STATIC_CREDENTIAL_PROVIDERS.AddProvider(self)

    def UnRegister(self):
        properties.VALUES.core.account.RemoveCallback(self.GetAccount)
        properties.VALUES.core.project.RemoveCallback(self.GetProject)
        properties.VALUES.core.universe_domain.RemoveCallback(self.GetUniverseDomain)
        STATIC_CREDENTIAL_PROVIDERS.RemoveProvider(self)