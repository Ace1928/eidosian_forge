from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class NoStagesError(exceptions.Error):
    """Error when a release doesn't contain any pipeline stages."""

    def __init__(self, release_name):
        super(NoStagesError, self).__init__('No pipeline stages in the release {}.'.format(release_name))