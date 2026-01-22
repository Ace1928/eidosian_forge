from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class FileNotFoundError(exceptions.Error):
    """File or directory that was supposed to exist didn't exist."""

    def __init__(self, path):
        super(FileNotFoundError, self).__init__('[{}] does not exist.'.format(path))