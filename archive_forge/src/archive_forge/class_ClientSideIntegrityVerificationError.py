from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class ClientSideIntegrityVerificationError(core_exceptions.Error):
    """Error raised when client identifies integrity verification error."""