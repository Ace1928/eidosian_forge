from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
class OsloginState(object):
    """Class for holding OS Login State.

  Attributes:
    oslogin_enabled: bool, True if OS Login is enabled on the instance.
    oslogin_2fa_enabled: bool, True if OS Login 2FA is enabled on the instance.
    security_keys_enabled: bool, True if Security Keys should be used for SSH
      authentication.
    user: str, The username that SSH should use for connecting.
    third_party_user: bool, True if the authenticated user is an external
      account user.
    ssh_security_key_support: bool, True if the SSH client supports security
      keys.
    environment: str, A hint about the current enviornment. ('ssh' or 'putty')
    security_keys: list, A list of 'private' keys associated with the security
      keys configured in the user's account.
    signed_ssh_key: bool, True if a valid signed ssh key exists.
    require_certificates: bool, True if passing a certificate is required.
  """

    def __init__(self, oslogin_enabled=False, oslogin_2fa_enabled=False, security_keys_enabled=False, user=None, third_party_user=False, ssh_security_key_support=None, environment=None, security_keys=None, signed_ssh_key=False, require_certificates=False):
        self.oslogin_enabled = oslogin_enabled
        self.oslogin_2fa_enabled = oslogin_2fa_enabled
        self.security_keys_enabled = security_keys_enabled
        self.user = user
        self.third_party_user = third_party_user
        self.ssh_security_key_support = ssh_security_key_support
        self.environment = environment
        if security_keys is None:
            self.security_keys = []
        else:
            self.security_keys = security_keys
        self.signed_ssh_key = signed_ssh_key
        self.require_certificates = require_certificates

    def __str__(self):
        return textwrap.dedent('        OS Login Enabled: {0}\n        2FA Enabled: {1}\n        Security Keys Enabled: {2}\n        Username: {3}\n        Third Party User: {4}\n        SSH Security Key Support: {5}\n        Environment: {6}\n        Security Keys:\n        {7}\n        Signed SSH Key: {8}\n        Require Certificates: {9}\n        ').format(self.oslogin_enabled, self.oslogin_2fa_enabled, self.security_keys_enabled, self.user, self.third_party_user, self.ssh_security_key_support, self.environment, '\n'.join(self.security_keys), self.signed_ssh_key, self.require_certificates)

    def __repr__(self):
        return 'OsloginState(oslogin_enabled={0}, oslogin_2fa_enabled={1}, security_keys_enabled={2}, user={3}, third_party_user={4}ssh_security_key_support={5}, environment={6}, security_keys={7}, signed_ssh_key={8}, require_certificates={9})'.format(self.oslogin_enabled, self.oslogin_2fa_enabled, self.security_keys_enabled, self.user, self.third_party_user, self.ssh_security_key_support, self.environment, self.security_keys, self.signed_ssh_key, self.require_certificates)