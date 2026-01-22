from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from six.moves.urllib import parse
import uritemplate
def _ErrorFromMatrixInFailedState(matrix):
    """Produces a human-readable error message from an invalid matrix."""
    messages = apis.GetMessagesModule('testing', 'v1')
    if matrix.state == messages.TestMatrix.StateValueValuesEnum.INVALID:
        return _ExtractInvalidMatrixDetails(matrix)
    return _GenericErrorMessage(matrix)