from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MultiDeployError(DeployError):
    """Indicates a failed attempt to deploy multiple image urls."""

    def __str__(self):
        return 'No more than one service may be deployed when using the image-url or appyaml flag'