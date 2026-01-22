from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class SafetyError(exceptions.Error):
    """For when a safety check is required, but redundent.

  If this is thrown it means some other check failed. For example, a required
  argparse argument should never be None - argparse should throw an error if it
  is not passed - but safety requires we rule out the None value in later code.
  """