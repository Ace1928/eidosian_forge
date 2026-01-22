from sys import version_info as _swig_python_version_info
import weakref
def AddDisjunction(self, *args):
    """
        Adds a disjunction constraint on the indices: exactly 'max_cardinality' of
        the indices are active. Start and end indices of any vehicle cannot be
        part of a disjunction.

        If a penalty is given, at most 'max_cardinality' of the indices can be
        active, and if less are active, 'penalty' is payed per inactive index.
        This is equivalent to adding the constraint:
            p + Sum(i)active[i] == max_cardinality
        where p is an integer variable, and the following cost to the cost
        function:
            p * penalty.
        'penalty' must be positive to make the disjunction optional; a negative
        penalty will force 'max_cardinality' indices of the disjunction to be
        performed, and therefore p == 0.
        Note: passing a vector with a single index will model an optional index
        with a penalty cost if it is not visited.
        """
    return _pywrapcp.RoutingModel_AddDisjunction(self, *args)