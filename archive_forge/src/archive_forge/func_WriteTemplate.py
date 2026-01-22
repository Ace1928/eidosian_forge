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
def WriteTemplate(self):
    """Write the credential file."""
    self.Clean()
    if self._cred_type == c_creds.P12_SERVICE_ACCOUNT_CREDS_NAME:
        cred = self.credentials
        key = cred._private_key_pkcs12
        password = cred._private_key_password
        files.WriteBinaryFileContents(self._p12_key_path, key, private=True)
        self._WriteFileContents(self._gsutil_path, '\n'.join(['[Credentials]', 'gs_service_client_id = {account}', 'gs_service_key_file = {key_file}', 'gs_service_key_file_password = {key_password}']).format(account=self.credentials.service_account_email, key_file=self._p12_key_path, key_password=password))
        return
    c_creds.ADC(self.credentials).DumpADCToFile(file_path=self._adc_path)
    if self._cred_type == c_creds.EXTERNAL_ACCOUNT_CREDS_NAME or self._cred_type == c_creds.EXTERNAL_ACCOUNT_USER_CREDS_NAME:
        self._WriteFileContents(self._gsutil_path, '\n'.join(['[Credentials]', 'gs_external_account_file = {external_account_file}']).format(external_account_file=self._adc_path))
    elif self._cred_type == c_creds.EXTERNAL_ACCOUNT_AUTHORIZED_USER_CREDS_NAME:
        self._WriteFileContents(self._gsutil_path, '\n'.join(['[Credentials]', 'gs_external_account_authorized_user_file = {external_account_file}']).format(external_account_file=self._adc_path))
    elif self._cred_type == c_creds.USER_ACCOUNT_CREDS_NAME:
        self._WriteFileContents(self._gsutil_path, '\n'.join(['[OAuth2]', 'client_id = {cid}', 'client_secret = {secret}', '', '[Credentials]', 'gs_oauth2_refresh_token = {token}']).format(cid=self.credentials.client_id, secret=self.credentials.client_secret, token=self.credentials.refresh_token))
    elif self._cred_type == c_creds.SERVICE_ACCOUNT_CREDS_NAME:
        self._WriteFileContents(self._gsutil_path, '\n'.join(['[Credentials]', 'gs_service_key_file = {key_file}']).format(key_file=self._adc_path))
    else:
        raise c_creds.CredentialFileSaveError('Unsupported credentials type {0}'.format(type(self.credentials)))