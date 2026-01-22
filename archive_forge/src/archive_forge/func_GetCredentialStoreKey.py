from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import logging
import os
import io
import six
import traceback
from apitools.base.py import credentials_lib
from apitools.base.py import exceptions as apitools_exceptions
from boto import config
from gslib.cred_types import CredTypes
from gslib.exception import CommandException
from gslib.impersonation_credentials import ImpersonationCredentials
from gslib.no_op_credentials import NoOpCredentials
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import GetCredentialStoreFilename
from gslib.utils.boto_util import GetGceCredentialCacheFilename
from gslib.utils.boto_util import GetGcsJsonApiVersion
from gslib.utils.constants import UTF8
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from oauth2client.client import HAS_CRYPTO
from oauth2client.contrib import devshell
from oauth2client.service_account import ServiceAccountCredentials
from google_reauth import reauth_creds
from oauth2client.contrib import multiprocess_file_storage
from six import BytesIO
def GetCredentialStoreKey(credentials, api_version):
    """Disambiguates a credential for caching in a credential store.

  Different credential types have different fields that identify them.  This
  function assembles relevant information in a string to be used as the key for
  accessing a credential.  Note that in addition to uniquely identifying the
  entity to which a credential corresponds, we must differentiate between two or
  more of that entity's credentials that have different attributes such that the
  credentials should not be treated as interchangeable, e.g. if they target
  different API versions (happens for developers targeting different test
  environments), have different private key IDs (for service account JSON
  keyfiles), or target different provider token (refresh) URIs.

  Args:
    credentials: An OAuth2Credentials object.
    api_version: JSON API version being used.

  Returns:
    A string that can be used as the key to identify a credential, e.g.
    "v1-909320924072.apps.googleusercontent.com-1/rEfrEshtOkEn-https://..."
  """
    key_parts = [api_version]
    if isinstance(credentials, devshell.DevshellCredentials):
        key_parts.append(credentials.user_email)
    elif isinstance(credentials, ServiceAccountCredentials):
        key_parts.append(credentials._service_account_email)
        if getattr(credentials, '_private_key_id', None):
            key_parts.append(credentials._private_key_id)
        elif getattr(credentials, '_private_key_pkcs12', None):
            key_parts.append(base64.b64encode(credentials._private_key_pkcs12)[:20])
    elif isinstance(credentials, oauth2client.client.OAuth2Credentials):
        if credentials.client_id and credentials.client_id != 'null':
            key_parts.append(credentials.client_id)
        else:
            key_parts.append('noclientid')
        key_parts.append(credentials.refresh_token or 'norefreshtoken')
    if getattr(credentials, 'token_uri', None):
        key_parts.append(credentials.token_uri)
    key_parts = [six.ensure_text(part) for part in key_parts]
    return '-'.join(key_parts)