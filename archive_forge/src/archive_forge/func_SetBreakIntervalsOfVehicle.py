from sys import version_info as _swig_python_version_info
import weakref
def SetBreakIntervalsOfVehicle(self, breaks, vehicle, node_visit_transits):
    """
        Sets the breaks for a given vehicle. Breaks are represented by
        IntervalVars. They may interrupt transits between nodes and increase
        the value of corresponding slack variables.
        A break may take place before the start of a vehicle, after the end of
        a vehicle, or during a travel i -> j.

        In that case, the interval [break.Start(), break.End()) must be a subset
        of [CumulVar(i) + pre_travel(i, j), CumulVar(j) - post_travel(i, j)). In
        other words, a break may not overlap any node n's visit, given by
        [CumulVar(n) - post_travel(_, n), CumulVar(n) + pre_travel(n, _)).
        This formula considers post_travel(_, start) and pre_travel(end, _) to be
        0; pre_travel will never be called on any (_, start) and post_travel will
        never we called on any (end, _). If pre_travel_evaluator or
        post_travel_evaluator is -1, it will be taken as a function that always
        returns 0.
        Deprecated, sets pre_travel(i, j) = node_visit_transit[i].
        """
    return _pywrapcp.RoutingDimension_SetBreakIntervalsOfVehicle(self, breaks, vehicle, node_visit_transits)