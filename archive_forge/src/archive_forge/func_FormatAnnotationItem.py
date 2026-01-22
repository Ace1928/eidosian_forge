from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def FormatAnnotationItem(self):
    """Render a secret path for the run.googleapis.com/secrets annotation.

    Returns:
      str like 'projects/123/secrets/s1'

    Raises:
      TypeError for a local secret name that doesn't belong in the annotation.
    """
    if not self._IsRemote():
        raise TypeError('Only remote paths go in annotations')
    return 'projects/{remote_project_number}/secrets/{secret_name}'.format(remote_project_number=self.remote_project_number, secret_name=self.secret_name)