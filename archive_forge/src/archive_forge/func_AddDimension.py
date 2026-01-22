from sys import version_info as _swig_python_version_info
import weakref
def AddDimension(self, evaluator_index, slack_max, capacity, fix_start_cumul_to_zero, name):
    """
        Model creation
        Methods to add dimensions to routes; dimensions represent quantities
        accumulated at nodes along the routes. They represent quantities such as
        weights or volumes carried along the route, or distance or times.
        Quantities at a node are represented by "cumul" variables and the increase
        or decrease of quantities between nodes are represented by "transit"
        variables. These variables are linked as follows:
        if j == next(i), cumul(j) = cumul(i) + transit(i, j) + slack(i)
        where slack is a positive slack variable (can represent waiting times for
        a time dimension).
        Setting the value of fix_start_cumul_to_zero to true will force the
        "cumul" variable of the start node of all vehicles to be equal to 0.
        Creates a dimension where the transit variable is constrained to be
        equal to evaluator(i, next(i)); 'slack_max' is the upper bound of the
        slack variable and 'capacity' is the upper bound of the cumul variables.
        'name' is the name used to reference the dimension; this name is used to
        get cumul and transit variables from the routing model.
        Returns false if a dimension with the same name has already been created
        (and doesn't create the new dimension).
        Takes ownership of the callback 'evaluator'.
        """
    return _pywrapcp.RoutingModel_AddDimension(self, evaluator_index, slack_max, capacity, fix_start_cumul_to_zero, name)