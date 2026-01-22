from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
def GetSshPublicKey(self, user, fingerprint):
    """Get an SSH public key from the user's login profile.

    Args:
      user: str, The email address of the OS Login user.
      fingerprint: str, The fingerprint of the SSH key to delete.
    Returns:
      The requested SSH public key.
    """
    message = self.messages.OsloginUsersSshPublicKeysGetRequest(name='users/{0}/sshPublicKeys/{1}'.format(user, fingerprint))
    res = self.client.users_sshPublicKeys.Get(message)
    return res