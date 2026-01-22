from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.docker import credential_utils as cred_utils
from googlecloudsdk.core.util import files as file_utils
def CheckValidRegistry(self, registry):
    if registry not in cred_utils.SupportedRegistries():
        log.warning('{0} is not a supported registry'.format(registry))
        return False
    return True