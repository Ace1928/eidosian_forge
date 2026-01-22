from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class GcloudStorageTranslationError(Exception):
    """Exception raised when a gsutil command can't be translated to gcloud."""
    pass