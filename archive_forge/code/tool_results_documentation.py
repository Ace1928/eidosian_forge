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
Converts legacy invalid matrix enum to a descriptive message for the user.

  Args:
    matrix: A TestMatrix in a failed state

  Returns:
    A string containing the legacy error message when no message is available
    from the API.

  