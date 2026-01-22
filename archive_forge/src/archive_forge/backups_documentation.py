from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
Deletes backup with the given name.

  Args:
    project: the project id to get backup, a string.
    location: the location to get backup, a string.
    backup: the backup id to get backup, a string.

  Returns:
    Empty.
  