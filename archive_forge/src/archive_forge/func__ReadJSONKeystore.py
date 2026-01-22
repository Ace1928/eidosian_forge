from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
import copy
from datetime import datetime
from datetime import timedelta
import getpass
import json
import re
import sys
import six
from six.moves import urllib
from apitools.base.py.exceptions import HttpError
from apitools.base.py.http_wrapper import MakeRequest
from apitools.base.py.http_wrapper import Request
from boto import config
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import constants
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.shim_util import GcloudStorageMap, GcloudStorageFlag
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
def _ReadJSONKeystore(ks_contents, passwd=None):
    """Read the client email and private key from a JSON keystore.

  Assumes this keystore was downloaded from the Cloud Platform Console.
  By default, JSON keystore private keys from the Cloud Platform Console
  aren't encrypted so the passwd is optional as load_privatekey will
  prompt for the PEM passphrase if the key is encrypted.

  Arguments:
    ks_contents: JSON formatted string representing the keystore contents. Must
                 be a valid JSON string and contain the fields 'private_key'
                 and 'client_email'.
    passwd: Passphrase for encrypted private keys.

  Returns:
    key: Parsed private key from the keystore.
    client_email: The email address for the service account.

  Raises:
    ValueError: If unable to parse ks_contents or keystore is missing
                required fields.
  """
    ks = json.loads(six.ensure_str(ks_contents))
    if 'client_email' not in ks or 'private_key' not in ks:
        raise ValueError("JSON keystore doesn't contain required fields")
    client_email = ks['client_email']
    if passwd:
        key = load_privatekey(FILETYPE_PEM, ks['private_key'], passwd)
    else:
        key = load_privatekey(FILETYPE_PEM, ks['private_key'])
    return (key, client_email)