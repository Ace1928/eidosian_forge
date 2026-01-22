from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def GetStep(self):
    """Return the step for this cluster.

    Returns:
      The step for this cluster. May be None if this is not a leaf.
    """
    return self.__step