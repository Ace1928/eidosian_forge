from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
def LegacyCredentialsDir(self, account):
    """Gets the path to store legacy credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the credentials file.
    """
    if not account:
        account = 'default'
    account = account.replace(':', '')
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS and (account.upper().startswith('CON.') or account.upper().startswith('PRN.') or account.upper().startswith('AUX.') or account.upper().startswith('NUL.')):
        account = '.' + account
    return os.path.join(self.global_config_dir, 'legacy_credentials', account)