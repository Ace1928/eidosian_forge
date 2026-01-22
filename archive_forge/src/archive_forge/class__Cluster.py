from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
class _Cluster(object):
    """Encapsulation of a single cluster in the final Step-Graph.

  The cluster hierarchy represents pieces of the user_name. A given cluster is
  either a leaf (it contains a single step and no sub-clusters) or a transform
  (it contains no step and one or more sub-clusters).
  """

    def __init__(self, parent, name_in_parent):
        self.__children = {}
        self.__parent = parent
        self.__name_in_parent = name_in_parent
        self.__step = None

    def IsLeaf(self):
        """A leaf cluster contains no sub-clusters.

    Returns:
      True iff this is a leaf cluster.
    """
        return not self.__children

    def IsSingleton(self):
        """A singleton is any cluster that contains a single child.

    Returns:
      True iff this is a singleton cluster.
    """
        return len(self.__children) == 1

    def IsRoot(self):
        """Determine if this cluster is the root.

    Returns:
      True iff this is the root cluster.
    """
        return not self.__parent

    def GetStep(self):
        """Return the step for this cluster.

    Returns:
      The step for this cluster. May be None if this is not a leaf.
    """
        return self.__step

    def SetStep(self, step):
        """Set the step for this cluster.

    Can only be called on leaf nodes that have not yet had their step set.

    Args:
      step: The step that this cluster represents.
    """
        assert not self.__children
        assert not self.__step
        self.__step = step

    def Name(self, relative_to=None):
        """Return the name of this sub-cluster relative to the given ancestor.

    Args:
      relative_to: The ancestor to output the name relative to.

    Returns:
      The string representing the user_name component for this cluster.
    """
        if not self.__parent or self.__parent == relative_to:
            return self.__name_in_parent
        parent_name = self.__parent.Name(relative_to)
        if parent_name:
            return parent_name + '/' + self.__name_in_parent
        else:
            return self.__name_in_parent

    def GetOrAddChild(self, piece_name):
        """Return the cluster representing the given piece_name within this cluster.

    Args:
      piece_name: String representing the piece name of the desired child.
    Returns:
      Cluster representing the child.
    """
        assert not self.__step
        if piece_name not in self.__children:
            self.__children[piece_name] = _Cluster(self, piece_name)
        return self.__children[piece_name]

    def Children(self):
        """Return the sub-clusters.

    Returns:
      Sorted list of pairs for the children in this cluster.
    """
        return sorted(six.iteritems(self.__children))