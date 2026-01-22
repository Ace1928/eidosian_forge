from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnsupportedIntegrationsLocationError(exceptions.Error):
    """An error encountered when an unsupported location is provided."""