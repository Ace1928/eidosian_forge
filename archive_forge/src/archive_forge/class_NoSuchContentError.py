from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class NoSuchContentError(exceptions.Error):
    """For when trying to configure unsupported or missing content.

  For instance, if the user attempts to install a bundle that doesn't exist or
  isn't supported, this error should be thrown.
  """