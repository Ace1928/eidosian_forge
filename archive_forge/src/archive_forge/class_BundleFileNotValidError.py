from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class BundleFileNotValidError(exceptions.Error):
    """Raised when a bundle file is not valid.

  The deploy command validates that the bundle file provided by the
  --bundle-file command line flag is a valid zip archive, and if not, raises
  this exception.
  """

    def __init__(self, bundle_file):
        msg = 'Bundle file is not a valid zip archive: {}'.format(bundle_file)
        super(BundleFileNotValidError, self).__init__(msg)