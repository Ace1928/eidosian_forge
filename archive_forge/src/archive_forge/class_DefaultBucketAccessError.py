from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class DefaultBucketAccessError(DeployError):
    """Indicates a failed attempt to access a project's default bucket."""

    def __init__(self, project):
        super(DefaultBucketAccessError, self).__init__()
        self.project = project

    def __str__(self):
        return 'Could not retrieve the default Google Cloud Storage bucket for [{a}]. Please try again or use the [bucket] argument.'.format(a=self.project)