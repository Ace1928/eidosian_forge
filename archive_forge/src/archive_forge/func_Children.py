from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def Children(self):
    """Return the sub-clusters.

    Returns:
      Sorted list of pairs for the children in this cluster.
    """
    return sorted(six.iteritems(self.__children))