from sys import version_info as _swig_python_version_info
import weakref
class PathOperator(IntVarLocalSearchOperator):
    """
    Base class of the local search operators dedicated to path modifications
    (a path is a set of nodes linked together by arcs).
    This family of neighborhoods supposes they are handling next variables
    representing the arcs (var[i] represents the node immediately after i on
    a path).
    Several services are provided:
    - arc manipulators (SetNext(), ReverseChain(), MoveChain())
    - path inspectors (Next(), Prev(), IsPathEnd())
    - path iterators: operators need a given number of nodes to define a
      neighbor; this class provides the iteration on a given number of (base)
      nodes which can be used to define a neighbor (through the BaseNode method)
    Subclasses only need to override MakeNeighbor to create neighbors using
    the services above (no direct manipulation of assignments).
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr

    def Neighbor(self):
        return _pywrapcp.PathOperator_Neighbor(self)