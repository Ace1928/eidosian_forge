from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class CloudDeployConfigError(exceptions.Error):
    """Error raised for errors in the cloud deploy yaml config."""