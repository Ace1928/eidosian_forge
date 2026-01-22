from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
def GetKeysFromProfile(user, oslogin_client):
    profile = oslogin_client.GetLoginProfile(user)
    if profile.sshPublicKeys:
        return profile.sshPublicKeys.additionalProperties