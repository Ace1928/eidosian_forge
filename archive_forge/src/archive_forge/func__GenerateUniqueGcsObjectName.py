from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def _GenerateUniqueGcsObjectName():
    """Create a unique GCS object name to hold test results in the results bucket.

  The Testing back-end needs a unique GCS object name within the results bucket
  to prevent race conditions while processing test results. By default, the
  gcloud client uses the current time down to the microsecond in ISO format plus
  a random 4-letter suffix. The format is: "YYYY-MM-DD_hh:mm:ss.ssssss_rrrr".

  Returns:
    A string with the unique GCS object name.
  """
    return '{0}_{1}'.format(datetime.datetime.now().isoformat(b'_' if six.PY2 else '_'), ''.join(random.sample(string.ascii_letters, 4)))