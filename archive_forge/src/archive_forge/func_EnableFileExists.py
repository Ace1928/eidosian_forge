from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def EnableFileExists(ve_dir):
    """Returns True if enable file exists."""
    return os.path.exists('{}/{}'.format(ve_dir, ENABLE_FILE))