from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
class InvalidDestinationError(Error):

    def __init__(self, source, dest):
        super(InvalidDestinationError, self).__init__('Cannot copy [{}] to [{}] because of "." or ".." in the path. gcloud does not support Cloud Storage paths containing these path segments and it is recommended that you do not name objects in this way. Other tooling may convert these paths to incorrect local directories.'.format(source.path, dest.path))