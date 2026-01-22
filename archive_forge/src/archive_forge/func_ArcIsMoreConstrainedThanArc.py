from sys import version_info as _swig_python_version_info
import weakref
def ArcIsMoreConstrainedThanArc(self, _from, to1, to2):
    """
        Returns whether the arc from->to1 is more constrained than from->to2,
        taking into account, in order:
        - whether the destination node isn't an end node
        - whether the destination node is mandatory
        - whether the destination node is bound to the same vehicle as the source
        - the "primary constrained" dimension (see SetPrimaryConstrainedDimension)
        It then breaks ties using, in order:
        - the arc cost (taking unperformed penalties into account)
        - the size of the vehicle vars of "to1" and "to2" (lowest size wins)
        - the value: the lowest value of the indices to1 and to2 wins.
        See the .cc for details.
        The more constrained arc is typically preferable when building a
        first solution. This method is intended to be used as a callback for the
        BestValueByComparisonSelector value selector.
        Args:
          from: the variable index of the source node
          to1: the variable index of the first candidate destination node.
          to2: the variable index of the second candidate destination node.
        """
    return _pywrapcp.RoutingModel_ArcIsMoreConstrainedThanArc(self, _from, to1, to2)