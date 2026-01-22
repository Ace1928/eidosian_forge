from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def GetResourceNameMismatchErrorMessage(request_resource_name, response_resource_name):
    return 'Found a mismatch between user-requested crypto resource ({})'.format(request_resource_name) + 'and server-reported resource used for the cryptographic operation ({}).\n'.format(response_resource_name) + _ERROR_MESSAGE_SUFFIX