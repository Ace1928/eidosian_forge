from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def IsSingleton(self):
    """A singleton is any cluster that contains a single child.

    Returns:
      True iff this is a singleton cluster.
    """
    return len(self.__children) == 1