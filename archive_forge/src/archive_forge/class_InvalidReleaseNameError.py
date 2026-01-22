from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidReleaseNameError(exceptions.Error):
    """Error when a release has extra $ signs after expanding template terms."""

    def __init__(self, release_name, error_indices):
        error_msg = "Invalid character '$' for release name '{}' at indices: {}. Did you mean to use $DATE or $TIME?"
        super(InvalidReleaseNameError, self).__init__(error_msg.format(release_name, error_indices))