from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
def GetKeyDictionaryFromProfile(user, oslogin_client, profile=None):
    """Return a dictionary of fingerprints/keys from the OS Login Profile."""
    if not profile:
        profile = oslogin_client.GetLoginProfile(user)
    key_dir = {}
    if not profile.sshPublicKeys:
        return {}
    for ssh_pub_key in profile.sshPublicKeys.additionalProperties:
        key_dir[ssh_pub_key.key] = ssh_pub_key.value.key
    return key_dir