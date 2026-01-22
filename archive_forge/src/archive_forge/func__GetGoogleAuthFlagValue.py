from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
import bootstrapping
from googlecloudsdk.api_lib.iamcredentials import util as iamcred_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce
from googlecloudsdk.core.credentials import store
def _GetGoogleAuthFlagValue(argv):
    for arg in argv[1:]:
        if re.fullmatch('--use_google_auth(=True)*', arg):
            return True
        if re.fullmatch('(--nouse_google_auth|--use_google_auth=False)', arg):
            return False
    return None