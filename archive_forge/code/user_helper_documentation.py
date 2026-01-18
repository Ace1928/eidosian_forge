from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
Validates command line arguments and creates the user and field mask.

  Args:
    alloydb_messages: Messages module for the API client.
    user_ref: resource path of the resource being updated
    args: Command line input arguments.

  Returns:
    An AlloyDB user and mask for update.
  