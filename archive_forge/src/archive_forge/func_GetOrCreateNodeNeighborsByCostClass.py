from sys import version_info as _swig_python_version_info
import weakref
def GetOrCreateNodeNeighborsByCostClass(self, *args):
    """
        *Overload 1:*
        Returns neighbors of all nodes for every cost class. The result is cached
        and is computed once. The number of neighbors considered is based on a
        ratio of non-vehicle nodes, specified by neighbors_ratio, with a minimum
        of min-neighbors node considered.

        |

        *Overload 2:*
        Returns parameters.num_neighbors neighbors of all nodes for every cost
        class. The result is cached and is computed once.

        |

        *Overload 3:*
        Returns parameters.num_neighbors neighbors of all nodes for every cost
        class. The result is cached and is computed once.
        """
    return _pywrapcp.RoutingModel_GetOrCreateNodeNeighborsByCostClass(self, *args)