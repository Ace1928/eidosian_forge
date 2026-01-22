from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def IsRoot(self):
    """Determine if this cluster is the root.

    Returns:
      True iff this is the root cluster.
    """
    return not self.__parent