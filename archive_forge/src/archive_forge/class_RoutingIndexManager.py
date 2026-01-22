from sys import version_info as _swig_python_version_info
import weakref
class RoutingIndexManager(object):
    """
    Manager for any NodeIndex <-> variable index conversion. The routing solver
    uses variable indices internally and through its API. These variable indices
    are tricky to manage directly because one Node can correspond to a multitude
    of variables, depending on the number of times they appear in the model, and
    if they're used as start and/or end points. This class aims to simplify
    variable index usage, allowing users to use NodeIndex instead.

    Usage:

      .. code-block:: c++

          auto starts_ends = ...;  /// These are NodeIndex.
          RoutingIndexManager manager(10, 4, starts_ends);  // 10 nodes, 4 vehicles.
          RoutingModel model(manager);

    Then, use 'manager.NodeToIndex(node)' whenever model requires a variable
    index.

    Note: the mapping between node indices and variables indices is subject to
    change so no assumption should be made on it. The only guarantee is that
    indices range between 0 and n-1, where n = number of vehicles * 2 (for start
    and end nodes) + number of non-start or end nodes.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        _pywrapcp.RoutingIndexManager_swiginit(self, _pywrapcp.new_RoutingIndexManager(*args))

    def GetNumberOfNodes(self):
        return _pywrapcp.RoutingIndexManager_GetNumberOfNodes(self)

    def GetNumberOfVehicles(self):
        return _pywrapcp.RoutingIndexManager_GetNumberOfVehicles(self)

    def GetNumberOfIndices(self):
        return _pywrapcp.RoutingIndexManager_GetNumberOfIndices(self)

    def GetStartIndex(self, vehicle):
        return _pywrapcp.RoutingIndexManager_GetStartIndex(self, vehicle)

    def GetEndIndex(self, vehicle):
        return _pywrapcp.RoutingIndexManager_GetEndIndex(self, vehicle)

    def NodeToIndex(self, node):
        return _pywrapcp.RoutingIndexManager_NodeToIndex(self, node)

    def IndexToNode(self, index):
        return _pywrapcp.RoutingIndexManager_IndexToNode(self, index)
    __swig_destroy__ = _pywrapcp.delete_RoutingIndexManager