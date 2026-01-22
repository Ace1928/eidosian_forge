from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def IsPy2():
    """Wrap six.PY2, needed because mocking six.PY2 breaks test lib things."""
    return six.PY2