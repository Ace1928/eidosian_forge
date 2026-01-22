from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class AbandonedReleaseError(exceptions.Error):
    """Error when an activity happens on an abandoned release."""

    def __init__(self, error_msg, release_name):
        error_template = '{} Release {} is abandoned.'.format(error_msg, release_name)
        super(AbandonedReleaseError, self).__init__(error_template)