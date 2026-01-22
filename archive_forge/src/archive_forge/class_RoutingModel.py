from sys import version_info as _swig_python_version_info
import weakref
class RoutingModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    ROUTING_NOT_SOLVED = _pywrapcp.RoutingModel_ROUTING_NOT_SOLVED
    ' Problem not solved yet (before calling RoutingModel::Solve()).'
    ROUTING_SUCCESS = _pywrapcp.RoutingModel_ROUTING_SUCCESS
    ' Problem solved successfully after calling RoutingModel::Solve().'
    ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED = _pywrapcp.RoutingModel_ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED
    '\n    Problem solved successfully after calling RoutingModel::Solve(), except\n    that a local optimum has not been reached. Leaving more time would allow\n    improving the solution.\n    '
    ROUTING_FAIL = _pywrapcp.RoutingModel_ROUTING_FAIL
    ' No solution found to the problem after calling RoutingModel::Solve().'
    ROUTING_FAIL_TIMEOUT = _pywrapcp.RoutingModel_ROUTING_FAIL_TIMEOUT
    ' Time limit reached before finding a solution with RoutingModel::Solve().'
    ROUTING_INVALID = _pywrapcp.RoutingModel_ROUTING_INVALID
    ' Model, model parameters or flags are not valid.'
    ROUTING_INFEASIBLE = _pywrapcp.RoutingModel_ROUTING_INFEASIBLE
    ' Problem proven to be infeasible.'
    ROUTING_OPTIMAL = _pywrapcp.RoutingModel_ROUTING_OPTIMAL
    ' Problem has been solved to optimality.'
    PICKUP_AND_DELIVERY_NO_ORDER = _pywrapcp.RoutingModel_PICKUP_AND_DELIVERY_NO_ORDER
    ' Any precedence is accepted.'
    PICKUP_AND_DELIVERY_LIFO = _pywrapcp.RoutingModel_PICKUP_AND_DELIVERY_LIFO
    ' Deliveries must be performed in reverse order of pickups.'
    PICKUP_AND_DELIVERY_FIFO = _pywrapcp.RoutingModel_PICKUP_AND_DELIVERY_FIFO
    ' Deliveries must be performed in the same order as pickups.'

    def __init__(self, *args):
        _pywrapcp.RoutingModel_swiginit(self, _pywrapcp.new_RoutingModel(*args))
    __swig_destroy__ = _pywrapcp.delete_RoutingModel
    kTransitEvaluatorSignUnknown = _pywrapcp.RoutingModel_kTransitEvaluatorSignUnknown
    kTransitEvaluatorSignPositiveOrZero = _pywrapcp.RoutingModel_kTransitEvaluatorSignPositiveOrZero
    kTransitEvaluatorSignNegativeOrZero = _pywrapcp.RoutingModel_kTransitEvaluatorSignNegativeOrZero

    def RegisterUnaryTransitVector(self, values):
        """
        Registers 'callback' and returns its index.
        The sign parameter allows to notify the solver that the callback only
        return values of the given sign. This can help the solver, but passing
        an incorrect sign may crash in non-opt compilation mode, and yield
        incorrect results in opt.
        """
        return _pywrapcp.RoutingModel_RegisterUnaryTransitVector(self, values)

    def RegisterUnaryTransitCallback(self, *args):
        return _pywrapcp.RoutingModel_RegisterUnaryTransitCallback(self, *args)

    def RegisterTransitMatrix(self, values):
        return _pywrapcp.RoutingModel_RegisterTransitMatrix(self, values)

    def RegisterTransitCallback(self, *args):
        return _pywrapcp.RoutingModel_RegisterTransitCallback(self, *args)

    def TransitCallback(self, callback_index):
        return _pywrapcp.RoutingModel_TransitCallback(self, callback_index)

    def UnaryTransitCallbackOrNull(self, callback_index):
        return _pywrapcp.RoutingModel_UnaryTransitCallbackOrNull(self, callback_index)

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

    def AddDimensionWithVehicleTransits(self, evaluator_indices, slack_max, capacity, fix_start_cumul_to_zero, name):
        return _pywrapcp.RoutingModel_AddDimensionWithVehicleTransits(self, evaluator_indices, slack_max, capacity, fix_start_cumul_to_zero, name)

    def AddDimensionWithVehicleCapacity(self, evaluator_index, slack_max, vehicle_capacities, fix_start_cumul_to_zero, name):
        return _pywrapcp.RoutingModel_AddDimensionWithVehicleCapacity(self, evaluator_index, slack_max, vehicle_capacities, fix_start_cumul_to_zero, name)

    def AddDimensionWithVehicleTransitAndCapacity(self, evaluator_indices, slack_max, vehicle_capacities, fix_start_cumul_to_zero, name):
        return _pywrapcp.RoutingModel_AddDimensionWithVehicleTransitAndCapacity(self, evaluator_indices, slack_max, vehicle_capacities, fix_start_cumul_to_zero, name)

    def AddConstantDimensionWithSlack(self, value, capacity, slack_max, fix_start_cumul_to_zero, name):
        """
        Creates a dimension where the transit variable is constrained to be
        equal to 'value'; 'capacity' is the upper bound of the cumul variables.
        'name' is the name used to reference the dimension; this name is used to
        get cumul and transit variables from the routing model.
        Returns a pair consisting of an index to the registered unary transit
        callback and a bool denoting whether the dimension has been created.
        It is false if a dimension with the same name has already been created
        (and doesn't create the new dimension but still register a new callback).
        """
        return _pywrapcp.RoutingModel_AddConstantDimensionWithSlack(self, value, capacity, slack_max, fix_start_cumul_to_zero, name)

    def AddConstantDimension(self, value, capacity, fix_start_cumul_to_zero, name):
        return _pywrapcp.RoutingModel_AddConstantDimension(self, value, capacity, fix_start_cumul_to_zero, name)

    def AddVectorDimension(self, values, capacity, fix_start_cumul_to_zero, name):
        """
        Creates a dimension where the transit variable is constrained to be
        equal to 'values[i]' for node i; 'capacity' is the upper bound of
        the cumul variables. 'name' is the name used to reference the dimension;
        this name is used to get cumul and transit variables from the routing
        model.
        Returns a pair consisting of an index to the registered unary transit
        callback and a bool denoting whether the dimension has been created.
        It is false if a dimension with the same name has already been created
        (and doesn't create the new dimension but still register a new callback).
        """
        return _pywrapcp.RoutingModel_AddVectorDimension(self, values, capacity, fix_start_cumul_to_zero, name)

    def AddMatrixDimension(self, values, capacity, fix_start_cumul_to_zero, name):
        """
        Creates a dimension where the transit variable is constrained to be
        equal to 'values[i][next(i)]' for node i; 'capacity' is the upper bound of
        the cumul variables. 'name' is the name used to reference the dimension;
        this name is used to get cumul and transit variables from the routing
        model.
        Returns a pair consisting of an index to the registered transit callback
        and a bool denoting whether the dimension has been created.
        It is false if a dimension with the same name has already been created
        (and doesn't create the new dimension but still register a new callback).
        """
        return _pywrapcp.RoutingModel_AddMatrixDimension(self, values, capacity, fix_start_cumul_to_zero, name)

    def GetAllDimensionNames(self):
        """ Outputs the names of all dimensions added to the routing engine."""
        return _pywrapcp.RoutingModel_GetAllDimensionNames(self)

    def GetDimensions(self):
        """ Returns all dimensions of the model."""
        return _pywrapcp.RoutingModel_GetDimensions(self)

    def GetDimensionsWithSoftOrSpanCosts(self):
        """ Returns dimensions with soft or vehicle span costs."""
        return _pywrapcp.RoutingModel_GetDimensionsWithSoftOrSpanCosts(self)

    def GetUnaryDimensions(self):
        """ Returns dimensions for which all transit evaluators are unary."""
        return _pywrapcp.RoutingModel_GetUnaryDimensions(self)

    def GetDimensionsWithGlobalCumulOptimizers(self):
        """ Returns the dimensions which have [global|local]_dimension_optimizers_."""
        return _pywrapcp.RoutingModel_GetDimensionsWithGlobalCumulOptimizers(self)

    def GetDimensionsWithLocalCumulOptimizers(self):
        return _pywrapcp.RoutingModel_GetDimensionsWithLocalCumulOptimizers(self)

    def HasGlobalCumulOptimizer(self, dimension):
        """ Returns whether the given dimension has global/local cumul optimizers."""
        return _pywrapcp.RoutingModel_HasGlobalCumulOptimizer(self, dimension)

    def HasLocalCumulOptimizer(self, dimension):
        return _pywrapcp.RoutingModel_HasLocalCumulOptimizer(self, dimension)

    def GetMutableGlobalCumulLPOptimizer(self, dimension):
        """
        Returns the global/local dimension cumul optimizer for a given dimension,
        or nullptr if there is none.
        """
        return _pywrapcp.RoutingModel_GetMutableGlobalCumulLPOptimizer(self, dimension)

    def GetMutableGlobalCumulMPOptimizer(self, dimension):
        return _pywrapcp.RoutingModel_GetMutableGlobalCumulMPOptimizer(self, dimension)

    def GetMutableLocalCumulLPOptimizer(self, dimension):
        return _pywrapcp.RoutingModel_GetMutableLocalCumulLPOptimizer(self, dimension)

    def GetMutableLocalCumulMPOptimizer(self, dimension):
        return _pywrapcp.RoutingModel_GetMutableLocalCumulMPOptimizer(self, dimension)

    def HasDimension(self, dimension_name):
        """ Returns true if a dimension exists for a given dimension name."""
        return _pywrapcp.RoutingModel_HasDimension(self, dimension_name)

    def GetDimensionOrDie(self, dimension_name):
        """ Returns a dimension from its name. Dies if the dimension does not exist."""
        return _pywrapcp.RoutingModel_GetDimensionOrDie(self, dimension_name)

    def GetMutableDimension(self, dimension_name):
        """
        Returns a dimension from its name. Returns nullptr if the dimension does
        not exist.
        """
        return _pywrapcp.RoutingModel_GetMutableDimension(self, dimension_name)

    def SetPrimaryConstrainedDimension(self, dimension_name):
        """
        Set the given dimension as "primary constrained". As of August 2013, this
        is only used by ArcIsMoreConstrainedThanArc().
        "dimension" must be the name of an existing dimension, or be empty, in
        which case there will not be a primary dimension after this call.
        """
        return _pywrapcp.RoutingModel_SetPrimaryConstrainedDimension(self, dimension_name)

    def GetPrimaryConstrainedDimension(self):
        """ Get the primary constrained dimension, or an empty string if it is unset."""
        return _pywrapcp.RoutingModel_GetPrimaryConstrainedDimension(self)

    def GetResourceGroup(self, rg_index):
        return _pywrapcp.RoutingModel_GetResourceGroup(self, rg_index)

    def GetDimensionResourceGroupIndices(self, dimension):
        """
        Returns the indices of resource groups for this dimension. This method can
        only be called after the model has been closed.
        """
        return _pywrapcp.RoutingModel_GetDimensionResourceGroupIndices(self, dimension)

    def GetDimensionResourceGroupIndex(self, dimension):
        """
        Returns the index of the resource group attached to the dimension.
        DCHECKS that there's exactly one resource group for this dimension.
        """
        return _pywrapcp.RoutingModel_GetDimensionResourceGroupIndex(self, dimension)

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

    def GetDisjunctionIndices(self, index):
        """ Returns the indices of the disjunctions to which an index belongs."""
        return _pywrapcp.RoutingModel_GetDisjunctionIndices(self, index)

    def GetDisjunctionPenalty(self, index):
        """ Returns the penalty of the node disjunction of index 'index'."""
        return _pywrapcp.RoutingModel_GetDisjunctionPenalty(self, index)

    def GetDisjunctionMaxCardinality(self, index):
        """
        Returns the maximum number of possible active nodes of the node
        disjunction of index 'index'.
        """
        return _pywrapcp.RoutingModel_GetDisjunctionMaxCardinality(self, index)

    def GetNumberOfDisjunctions(self):
        """ Returns the number of node disjunctions in the model."""
        return _pywrapcp.RoutingModel_GetNumberOfDisjunctions(self)

    def HasMandatoryDisjunctions(self):
        """
        Returns true if the model contains mandatory disjunctions (ones with
        kNoPenalty as penalty).
        """
        return _pywrapcp.RoutingModel_HasMandatoryDisjunctions(self)

    def HasMaxCardinalityConstrainedDisjunctions(self):
        """
        Returns true if the model contains at least one disjunction which is
        constrained by its max_cardinality.
        """
        return _pywrapcp.RoutingModel_HasMaxCardinalityConstrainedDisjunctions(self)

    def GetPerfectBinaryDisjunctions(self):
        """
        Returns the list of all perfect binary disjunctions, as pairs of variable
        indices: a disjunction is "perfect" when its variables do not appear in
        any other disjunction. Each pair is sorted (lowest variable index first),
        and the output vector is also sorted (lowest pairs first).
        """
        return _pywrapcp.RoutingModel_GetPerfectBinaryDisjunctions(self)

    def IgnoreDisjunctionsAlreadyForcedToZero(self):
        """
        SPECIAL: Makes the solver ignore all the disjunctions whose active
        variables are all trivially zero (i.e. Max() == 0), by setting their
        max_cardinality to 0.
        This can be useful when using the BaseBinaryDisjunctionNeighborhood
        operators, in the context of arc-based routing.
        """
        return _pywrapcp.RoutingModel_IgnoreDisjunctionsAlreadyForcedToZero(self)

    def AddSoftSameVehicleConstraint(self, indices, cost):
        """
        Adds a soft constraint to force a set of variable indices to be on the
        same vehicle. If all nodes are not on the same vehicle, each extra vehicle
        used adds 'cost' to the cost function.
        """
        return _pywrapcp.RoutingModel_AddSoftSameVehicleConstraint(self, indices, cost)

    def SetAllowedVehiclesForIndex(self, vehicles, index):
        """
        Sets the vehicles which can visit a given node. If the node is in a
        disjunction, this will not prevent it from being unperformed.
        Specifying an empty vector of vehicles has no effect (all vehicles
        will be allowed to visit the node).
        """
        return _pywrapcp.RoutingModel_SetAllowedVehiclesForIndex(self, vehicles, index)

    def IsVehicleAllowedForIndex(self, vehicle, index):
        """ Returns true if a vehicle is allowed to visit a given node."""
        return _pywrapcp.RoutingModel_IsVehicleAllowedForIndex(self, vehicle, index)

    def AddPickupAndDelivery(self, pickup, delivery):
        """
        Notifies that index1 and index2 form a pair of nodes which should belong
        to the same route. This methods helps the search find better solutions,
        especially in the local search phase.
        It should be called each time you have an equality constraint linking
        the vehicle variables of two node (including for instance pickup and
        delivery problems):
            Solver* const solver = routing.solver();
            int64_t index1 = manager.NodeToIndex(node1);
            int64_t index2 = manager.NodeToIndex(node2);
            solver->AddConstraint(solver->MakeEquality(
                routing.VehicleVar(index1),
                routing.VehicleVar(index2)));
            routing.AddPickupAndDelivery(index1, index2);
        """
        return _pywrapcp.RoutingModel_AddPickupAndDelivery(self, pickup, delivery)

    def AddPickupAndDeliverySets(self, pickup_disjunction, delivery_disjunction):
        """
        Same as AddPickupAndDelivery but notifying that the performed node from
        the disjunction of index 'pickup_disjunction' is on the same route as the
        performed node from the disjunction of index 'delivery_disjunction'.
        """
        return _pywrapcp.RoutingModel_AddPickupAndDeliverySets(self, pickup_disjunction, delivery_disjunction)

    def GetPickupPositions(self, node_index):
        """ Returns the pickup and delivery positions where the node is a pickup."""
        return _pywrapcp.RoutingModel_GetPickupPositions(self, node_index)

    def GetDeliveryPositions(self, node_index):
        """ Returns the pickup and delivery positions where the node is a delivery."""
        return _pywrapcp.RoutingModel_GetDeliveryPositions(self, node_index)

    def IsPickup(self, node_index):
        """ Returns whether the node is a pickup (resp. delivery)."""
        return _pywrapcp.RoutingModel_IsPickup(self, node_index)

    def IsDelivery(self, node_index):
        return _pywrapcp.RoutingModel_IsDelivery(self, node_index)

    def SetPickupAndDeliveryPolicyOfAllVehicles(self, policy):
        """
        Sets the Pickup and delivery policy of all vehicles. It is equivalent to
        calling SetPickupAndDeliveryPolicyOfVehicle on all vehicles.
        """
        return _pywrapcp.RoutingModel_SetPickupAndDeliveryPolicyOfAllVehicles(self, policy)

    def SetPickupAndDeliveryPolicyOfVehicle(self, policy, vehicle):
        return _pywrapcp.RoutingModel_SetPickupAndDeliveryPolicyOfVehicle(self, policy, vehicle)

    def GetPickupAndDeliveryPolicyOfVehicle(self, vehicle):
        return _pywrapcp.RoutingModel_GetPickupAndDeliveryPolicyOfVehicle(self, vehicle)

    def GetNumOfSingletonNodes(self):
        """
        Returns the number of non-start/end nodes which do not appear in a
        pickup/delivery pair.
        """
        return _pywrapcp.RoutingModel_GetNumOfSingletonNodes(self)
    TYPE_ADDED_TO_VEHICLE = _pywrapcp.RoutingModel_TYPE_ADDED_TO_VEHICLE
    " When visited, the number of types 'T' on the vehicle increases by one."
    ADDED_TYPE_REMOVED_FROM_VEHICLE = _pywrapcp.RoutingModel_ADDED_TYPE_REMOVED_FROM_VEHICLE
    "\n    When visited, one instance of type 'T' previously added to the route\n    (TYPE_ADDED_TO_VEHICLE), if any, is removed from the vehicle.\n    If the type was not previously added to the route or all added instances\n    have already been removed, this visit has no effect on the types.\n    "
    TYPE_ON_VEHICLE_UP_TO_VISIT = _pywrapcp.RoutingModel_TYPE_ON_VEHICLE_UP_TO_VISIT
    "\n    With the following policy, the visit enforces that type 'T' is\n    considered on the route from its start until this node is visited.\n    "
    TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED = _pywrapcp.RoutingModel_TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED
    "\n    The visit doesn't have an impact on the number of types 'T' on the\n    route, as it's (virtually) added and removed directly.\n    This policy can be used for visits which are part of an incompatibility\n    or requirement set without affecting the type count on the route.\n    "

    def SetVisitType(self, index, type, type_policy):
        return _pywrapcp.RoutingModel_SetVisitType(self, index, type, type_policy)

    def GetVisitType(self, index):
        return _pywrapcp.RoutingModel_GetVisitType(self, index)

    def GetSingleNodesOfType(self, type):
        return _pywrapcp.RoutingModel_GetSingleNodesOfType(self, type)

    def GetPairIndicesOfType(self, type):
        return _pywrapcp.RoutingModel_GetPairIndicesOfType(self, type)

    def GetVisitTypePolicy(self, index):
        return _pywrapcp.RoutingModel_GetVisitTypePolicy(self, index)

    def CloseVisitTypes(self):
        """
        This function should be called once all node visit types have been set and
        prior to adding any incompatibilities/requirements.
        "close" types.
        """
        return _pywrapcp.RoutingModel_CloseVisitTypes(self)

    def GetNumberOfVisitTypes(self):
        return _pywrapcp.RoutingModel_GetNumberOfVisitTypes(self)

    def AddHardTypeIncompatibility(self, type1, type2):
        """
        Incompatibilities:
        Two nodes with "hard" incompatible types cannot share the same route at
        all, while with a "temporal" incompatibility they can't be on the same
        route at the same time.
        """
        return _pywrapcp.RoutingModel_AddHardTypeIncompatibility(self, type1, type2)

    def AddTemporalTypeIncompatibility(self, type1, type2):
        return _pywrapcp.RoutingModel_AddTemporalTypeIncompatibility(self, type1, type2)

    def GetHardTypeIncompatibilitiesOfType(self, type):
        """ Returns visit types incompatible with a given type."""
        return _pywrapcp.RoutingModel_GetHardTypeIncompatibilitiesOfType(self, type)

    def GetTemporalTypeIncompatibilitiesOfType(self, type):
        return _pywrapcp.RoutingModel_GetTemporalTypeIncompatibilitiesOfType(self, type)

    def HasHardTypeIncompatibilities(self):
        """
        Returns true iff any hard (resp. temporal) type incompatibilities have
        been added to the model.
        """
        return _pywrapcp.RoutingModel_HasHardTypeIncompatibilities(self)

    def HasTemporalTypeIncompatibilities(self):
        return _pywrapcp.RoutingModel_HasTemporalTypeIncompatibilities(self)

    def AddSameVehicleRequiredTypeAlternatives(self, dependent_type, required_type_alternatives):
        """
        Requirements:
        NOTE: As of 2019-04, cycles in the requirement graph are not supported,
        and lead to the dependent nodes being skipped if possible (otherwise
        the model is considered infeasible).
        The following functions specify that "dependent_type" requires at least
        one of the types in "required_type_alternatives".

        For same-vehicle requirements, a node of dependent type type_D requires at
        least one node of type type_R among the required alternatives on the same
        route.
        """
        return _pywrapcp.RoutingModel_AddSameVehicleRequiredTypeAlternatives(self, dependent_type, required_type_alternatives)

    def AddRequiredTypeAlternativesWhenAddingType(self, dependent_type, required_type_alternatives):
        """
        If type_D depends on type_R when adding type_D, any node_D of type_D and
        VisitTypePolicy TYPE_ADDED_TO_VEHICLE or
        TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED requires at least one type_R on its
        vehicle at the time node_D is visited.
        """
        return _pywrapcp.RoutingModel_AddRequiredTypeAlternativesWhenAddingType(self, dependent_type, required_type_alternatives)

    def AddRequiredTypeAlternativesWhenRemovingType(self, dependent_type, required_type_alternatives):
        """
        The following requirements apply when visiting dependent nodes that remove
        their type from the route, i.e. type_R must be on the vehicle when type_D
        of VisitTypePolicy ADDED_TYPE_REMOVED_FROM_VEHICLE,
        TYPE_ON_VEHICLE_UP_TO_VISIT or TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED is
        visited.
        """
        return _pywrapcp.RoutingModel_AddRequiredTypeAlternativesWhenRemovingType(self, dependent_type, required_type_alternatives)

    def GetSameVehicleRequiredTypeAlternativesOfType(self, type):
        """
        Returns the set of same-vehicle requirement alternatives for the given
        type.
        """
        return _pywrapcp.RoutingModel_GetSameVehicleRequiredTypeAlternativesOfType(self, type)

    def GetRequiredTypeAlternativesWhenAddingType(self, type):
        """ Returns the set of requirement alternatives when adding the given type."""
        return _pywrapcp.RoutingModel_GetRequiredTypeAlternativesWhenAddingType(self, type)

    def GetRequiredTypeAlternativesWhenRemovingType(self, type):
        """ Returns the set of requirement alternatives when removing the given type."""
        return _pywrapcp.RoutingModel_GetRequiredTypeAlternativesWhenRemovingType(self, type)

    def HasSameVehicleTypeRequirements(self):
        """
        Returns true iff any same-route (resp. temporal) type requirements have
        been added to the model.
        """
        return _pywrapcp.RoutingModel_HasSameVehicleTypeRequirements(self)

    def HasTemporalTypeRequirements(self):
        return _pywrapcp.RoutingModel_HasTemporalTypeRequirements(self)

    def HasTypeRegulations(self):
        """
        Returns true iff the model has any incompatibilities or requirements set
        on node types.
        """
        return _pywrapcp.RoutingModel_HasTypeRegulations(self)

    def UnperformedPenalty(self, var_index):
        """
        Get the "unperformed" penalty of a node. This is only well defined if the
        node is only part of a single Disjunction, and that disjunction has a
        penalty. For forced active nodes returns max int64_t. In all other cases,
        this returns 0.
        """
        return _pywrapcp.RoutingModel_UnperformedPenalty(self, var_index)

    def UnperformedPenaltyOrValue(self, default_value, var_index):
        """
        Same as above except that it returns default_value instead of 0 when
        penalty is not well defined (default value is passed as first argument to
        simplify the usage of the method in a callback).
        """
        return _pywrapcp.RoutingModel_UnperformedPenaltyOrValue(self, default_value, var_index)

    def GetDepot(self):
        """
        Returns the variable index of the first starting or ending node of all
        routes. If all routes start  and end at the same node (single depot), this
        is the node returned.
        """
        return _pywrapcp.RoutingModel_GetDepot(self)

    def SetMaximumNumberOfActiveVehicles(self, max_active_vehicles):
        """
        Constrains the maximum number of active vehicles, aka the number of
        vehicles which do not have an empty route. For instance, this can be used
        to limit the number of routes in the case where there are fewer drivers
        than vehicles and that the fleet of vehicle is heterogeneous.
        """
        return _pywrapcp.RoutingModel_SetMaximumNumberOfActiveVehicles(self, max_active_vehicles)

    def GetMaximumNumberOfActiveVehicles(self):
        """ Returns the maximum number of active vehicles."""
        return _pywrapcp.RoutingModel_GetMaximumNumberOfActiveVehicles(self)

    def SetArcCostEvaluatorOfAllVehicles(self, evaluator_index):
        """
        Sets the cost function of the model such that the cost of a segment of a
        route between node 'from' and 'to' is evaluator(from, to), whatever the
        route or vehicle performing the route.
        """
        return _pywrapcp.RoutingModel_SetArcCostEvaluatorOfAllVehicles(self, evaluator_index)

    def SetArcCostEvaluatorOfVehicle(self, evaluator_index, vehicle):
        """ Sets the cost function for a given vehicle route."""
        return _pywrapcp.RoutingModel_SetArcCostEvaluatorOfVehicle(self, evaluator_index, vehicle)

    def SetFixedCostOfAllVehicles(self, cost):
        """
        Sets the fixed cost of all vehicle routes. It is equivalent to calling
        SetFixedCostOfVehicle on all vehicle routes.
        """
        return _pywrapcp.RoutingModel_SetFixedCostOfAllVehicles(self, cost)

    def SetFixedCostOfVehicle(self, cost, vehicle):
        """ Sets the fixed cost of one vehicle route."""
        return _pywrapcp.RoutingModel_SetFixedCostOfVehicle(self, cost, vehicle)

    def GetFixedCostOfVehicle(self, vehicle):
        """
        Returns the route fixed cost taken into account if the route of the
        vehicle is not empty, aka there's at least one node on the route other
        than the first and last nodes.
        """
        return _pywrapcp.RoutingModel_GetFixedCostOfVehicle(self, vehicle)

    def SetPathEnergyCostOfVehicle(self, force, distance, unit_cost, vehicle):
        return _pywrapcp.RoutingModel_SetPathEnergyCostOfVehicle(self, force, distance, unit_cost, vehicle)

    def SetAmortizedCostFactorsOfAllVehicles(self, linear_cost_factor, quadratic_cost_factor):
        """
        The following methods set the linear and quadratic cost factors of
        vehicles (must be positive values). The default value of these parameters
        is zero for all vehicles.

        When set, the cost_ of the model will contain terms aiming at reducing the
        number of vehicles used in the model, by adding the following to the
        objective for every vehicle v:
        INDICATOR(v used in the model) *
          [linear_cost_factor_of_vehicle_[v]
           - quadratic_cost_factor_of_vehicle_[v]*(square of length of route v)]
        i.e. for every used vehicle, we add the linear factor as fixed cost, and
        subtract the square of the route length multiplied by the quadratic
        factor. This second term aims at making the routes as dense as possible.

        Sets the linear and quadratic cost factor of all vehicles.
        """
        return _pywrapcp.RoutingModel_SetAmortizedCostFactorsOfAllVehicles(self, linear_cost_factor, quadratic_cost_factor)

    def SetAmortizedCostFactorsOfVehicle(self, linear_cost_factor, quadratic_cost_factor, vehicle):
        """ Sets the linear and quadratic cost factor of the given vehicle."""
        return _pywrapcp.RoutingModel_SetAmortizedCostFactorsOfVehicle(self, linear_cost_factor, quadratic_cost_factor, vehicle)

    def GetAmortizedLinearCostFactorOfVehicles(self):
        return _pywrapcp.RoutingModel_GetAmortizedLinearCostFactorOfVehicles(self)

    def GetAmortizedQuadraticCostFactorOfVehicles(self):
        return _pywrapcp.RoutingModel_GetAmortizedQuadraticCostFactorOfVehicles(self)

    def SetVehicleUsedWhenEmpty(self, is_used, vehicle):
        return _pywrapcp.RoutingModel_SetVehicleUsedWhenEmpty(self, is_used, vehicle)

    def IsVehicleUsedWhenEmpty(self, vehicle):
        return _pywrapcp.RoutingModel_IsVehicleUsedWhenEmpty(self, vehicle)

    def SetFirstSolutionEvaluator(self, evaluator):
        """
        Gets/sets the evaluator used during the search. Only relevant when
        RoutingSearchParameters.first_solution_strategy = EVALUATOR_STRATEGY.
        Takes ownership of evaluator.
        """
        return _pywrapcp.RoutingModel_SetFirstSolutionEvaluator(self, evaluator)

    def AddLocalSearchOperator(self, ls_operator):
        """
        Adds a local search operator to the set of operators used to solve the
        vehicle routing problem.
        """
        return _pywrapcp.RoutingModel_AddLocalSearchOperator(self, ls_operator)

    def AddSearchMonitor(self, monitor):
        """ Adds a search monitor to the search used to solve the routing model."""
        return _pywrapcp.RoutingModel_AddSearchMonitor(self, monitor)

    def AddAtSolutionCallback(self, callback, track_unchecked_neighbors=False):
        """
        Adds a callback called each time a solution is found during the search.
        This is a shortcut to creating a monitor to call the callback on
        AtSolution() and adding it with AddSearchMonitor.
        If track_unchecked_neighbors is true, the callback will also be called on
        AcceptUncheckedNeighbor() events, which is useful to grab solutions
        obtained when solver_parameters.check_solution_period > 1 (aka fastLS).
        """
        return _pywrapcp.RoutingModel_AddAtSolutionCallback(self, callback, track_unchecked_neighbors)

    def AddVariableMinimizedByFinalizer(self, var):
        """
        Adds a variable to minimize in the solution finalizer. The solution
        finalizer is called each time a solution is found during the search and
        allows to instantiate secondary variables (such as dimension cumul
        variables).
        """
        return _pywrapcp.RoutingModel_AddVariableMinimizedByFinalizer(self, var)

    def AddVariableMaximizedByFinalizer(self, var):
        """
        Adds a variable to maximize in the solution finalizer (see above for
        information on the solution finalizer).
        """
        return _pywrapcp.RoutingModel_AddVariableMaximizedByFinalizer(self, var)

    def AddWeightedVariableMinimizedByFinalizer(self, var, cost):
        """
        Adds a variable to minimize in the solution finalizer, with a weighted
        priority: the higher the more priority it has.
        """
        return _pywrapcp.RoutingModel_AddWeightedVariableMinimizedByFinalizer(self, var, cost)

    def AddWeightedVariableMaximizedByFinalizer(self, var, cost):
        """
        Adds a variable to maximize in the solution finalizer, with a weighted
        priority: the higher the more priority it has.
        """
        return _pywrapcp.RoutingModel_AddWeightedVariableMaximizedByFinalizer(self, var, cost)

    def AddVariableTargetToFinalizer(self, var, target):
        """
        Add a variable to set the closest possible to the target value in the
        solution finalizer.
        """
        return _pywrapcp.RoutingModel_AddVariableTargetToFinalizer(self, var, target)

    def AddWeightedVariableTargetToFinalizer(self, var, target, cost):
        """
        Same as above with a weighted priority: the higher the cost, the more
        priority it has to be set close to the target value.
        """
        return _pywrapcp.RoutingModel_AddWeightedVariableTargetToFinalizer(self, var, target, cost)

    def CloseModel(self):
        """
        Closes the current routing model; after this method is called, no
        modification to the model can be done, but RoutesToAssignment becomes
        available. Note that CloseModel() is automatically called by Solve() and
        other methods that produce solution.
        This is equivalent to calling
        CloseModelWithParameters(DefaultRoutingSearchParameters()).
        """
        return _pywrapcp.RoutingModel_CloseModel(self)

    def CloseModelWithParameters(self, search_parameters):
        """
        Same as above taking search parameters (as of 10/2015 some the parameters
        have to be set when closing the model).
        """
        return _pywrapcp.RoutingModel_CloseModelWithParameters(self, search_parameters)

    def Solve(self, assignment=None):
        """
        Solves the current routing model; closes the current model.
        This is equivalent to calling
        SolveWithParameters(DefaultRoutingSearchParameters())
        or
        SolveFromAssignmentWithParameters(assignment,
                                          DefaultRoutingSearchParameters()).
        """
        return _pywrapcp.RoutingModel_Solve(self, assignment)

    def SolveWithParameters(self, search_parameters, solutions=None):
        """
        Solves the current routing model with the given parameters. If 'solutions'
        is specified, it will contain the k best solutions found during the search
        (from worst to best, including the one returned by this method), where k
        corresponds to the 'number_of_solutions_to_collect' in
        'search_parameters'. Note that the Assignment returned by the method and
        the ones in solutions are owned by the underlying solver and should not be
        deleted.
        """
        return _pywrapcp.RoutingModel_SolveWithParameters(self, search_parameters, solutions)

    def SolveFromAssignmentWithParameters(self, assignment, search_parameters, solutions=None):
        """
        Same as above, except that if assignment is not null, it will be used as
        the initial solution.
        """
        return _pywrapcp.RoutingModel_SolveFromAssignmentWithParameters(self, assignment, search_parameters, solutions)

    def FastSolveFromAssignmentWithParameters(self, assignment, search_parameters, check_solution_in_cp, touched=None):
        """
        Improves a given assignment using unchecked local search.
        If check_solution_in_cp is true the final solution will be checked with
        the CP solver.
        As of 11/2023, only works with greedy descent.
        """
        return _pywrapcp.RoutingModel_FastSolveFromAssignmentWithParameters(self, assignment, search_parameters, check_solution_in_cp, touched)

    def SolveFromAssignmentsWithParameters(self, assignments, search_parameters, solutions=None):
        """
        Same as above but will try all assignments in order as first solutions
        until one succeeds.
        """
        return _pywrapcp.RoutingModel_SolveFromAssignmentsWithParameters(self, assignments, search_parameters, solutions)

    def SolveWithIteratedLocalSearch(self, search_parameters):
        """
        Solves the current routing model by using an Iterated Local Search
        approach.
        """
        return _pywrapcp.RoutingModel_SolveWithIteratedLocalSearch(self, search_parameters)

    def SetAssignmentFromOtherModelAssignment(self, target_assignment, source_model, source_assignment):
        """
        Given a "source_model" and its "source_assignment", resets
        "target_assignment" with the IntVar variables (nexts_, and vehicle_vars_
        if costs aren't homogeneous across vehicles) of "this" model, with the
        values set according to those in "other_assignment".
        The objective_element of target_assignment is set to this->cost_.
        """
        return _pywrapcp.RoutingModel_SetAssignmentFromOtherModelAssignment(self, target_assignment, source_model, source_assignment)

    def ComputeLowerBound(self):
        """
        Computes a lower bound to the routing problem solving a linear assignment
        problem. The routing model must be closed before calling this method.
        Note that problems with node disjunction constraints (including optional
        nodes) and non-homogenous costs are not supported (the method returns 0 in
        these cases).
        """
        return _pywrapcp.RoutingModel_ComputeLowerBound(self)

    def objective_lower_bound(self):
        """
        Returns the current lower bound found by internal solvers during the
        search.
        """
        return _pywrapcp.RoutingModel_objective_lower_bound(self)

    def status(self):
        """ Returns the current status of the routing model."""
        return _pywrapcp.RoutingModel_status(self)

    def enable_deep_serialization(self):
        """ Returns the value of the internal enable_deep_serialization_ parameter."""
        return _pywrapcp.RoutingModel_enable_deep_serialization(self)

    def ApplyLocks(self, locks):
        """
        Applies a lock chain to the next search. 'locks' represents an ordered
        vector of nodes representing a partial route which will be fixed during
        the next search; it will constrain next variables such that:
        next[locks[i]] == locks[i+1].

        Returns the next variable at the end of the locked chain; this variable is
        not locked. An assignment containing the locks can be obtained by calling
        PreAssignment().
        """
        return _pywrapcp.RoutingModel_ApplyLocks(self, locks)

    def ApplyLocksToAllVehicles(self, locks, close_routes):
        """
        Applies lock chains to all vehicles to the next search, such that locks[p]
        is the lock chain for route p. Returns false if the locks do not contain
        valid routes; expects that the routes do not contain the depots,
        i.e. there are empty vectors in place of empty routes.
        If close_routes is set to true, adds the end nodes to the route of each
        vehicle and deactivates other nodes.
        An assignment containing the locks can be obtained by calling
        PreAssignment().
        """
        return _pywrapcp.RoutingModel_ApplyLocksToAllVehicles(self, locks, close_routes)

    def PreAssignment(self):
        """
        Returns an assignment used to fix some of the variables of the problem.
        In practice, this assignment locks partial routes of the problem. This
        can be used in the context of locking the parts of the routes which have
        already been driven in online routing problems.
        """
        return _pywrapcp.RoutingModel_PreAssignment(self)

    def MutablePreAssignment(self):
        return _pywrapcp.RoutingModel_MutablePreAssignment(self)

    def WriteAssignment(self, file_name):
        """
        Writes the current solution to a file containing an AssignmentProto.
        Returns false if the file cannot be opened or if there is no current
        solution.
        """
        return _pywrapcp.RoutingModel_WriteAssignment(self, file_name)

    def ReadAssignment(self, file_name):
        """
        Reads an assignment from a file and returns the current solution.
        Returns nullptr if the file cannot be opened or if the assignment is not
        valid.
        """
        return _pywrapcp.RoutingModel_ReadAssignment(self, file_name)

    def RestoreAssignment(self, solution):
        """
        Restores an assignment as a solution in the routing model and returns the
        new solution. Returns nullptr if the assignment is not valid.
        """
        return _pywrapcp.RoutingModel_RestoreAssignment(self, solution)

    def ReadAssignmentFromRoutes(self, routes, ignore_inactive_indices):
        """
        Restores the routes as the current solution. Returns nullptr if the
        solution cannot be restored (routes do not contain a valid solution). Note
        that calling this method will run the solver to assign values to the
        dimension variables; this may take considerable amount of time, especially
        when using dimensions with slack.
        """
        return _pywrapcp.RoutingModel_ReadAssignmentFromRoutes(self, routes, ignore_inactive_indices)

    def RoutesToAssignment(self, routes, ignore_inactive_indices, close_routes, assignment):
        """
        Fills an assignment from a specification of the routes of the
        vehicles. The routes are specified as lists of variable indices that
        appear on the routes of the vehicles. The indices of the outer vector in
        'routes' correspond to vehicles IDs, the inner vector contains the
        variable indices on the routes for the given vehicle. The inner vectors
        must not contain the start and end indices, as these are determined by the
        routing model.  Sets the value of NextVars in the assignment, adding the
        variables to the assignment if necessary. The method does not touch other
        variables in the assignment. The method can only be called after the model
        is closed.  With ignore_inactive_indices set to false, this method will
        fail (return nullptr) in case some of the route contain indices that are
        deactivated in the model; when set to true, these indices will be
        skipped.  Returns true if routes were successfully
        loaded. However, such assignment still might not be a valid
        solution to the routing problem due to more complex constraints;
        it is advisible to call solver()->CheckSolution() afterwards.
        """
        return _pywrapcp.RoutingModel_RoutesToAssignment(self, routes, ignore_inactive_indices, close_routes, assignment)

    def AssignmentToRoutes(self, assignment, routes):
        """
        Converts the solution in the given assignment to routes for all vehicles.
        Expects that assignment contains a valid solution (i.e. routes for all
        vehicles end with an end index for that vehicle).
        """
        return _pywrapcp.RoutingModel_AssignmentToRoutes(self, assignment, routes)

    def CompactAssignment(self, assignment):
        """
        Converts the solution in the given assignment to routes for all vehicles.
        If the returned vector is route_indices, route_indices[i][j] is the index
        for jth location visited on route i. Note that contrary to
        AssignmentToRoutes, the vectors do include start and end locations.
        Returns a compacted version of the given assignment, in which all vehicles
        with id lower or equal to some N have non-empty routes, and all vehicles
        with id greater than N have empty routes. Does not take ownership of the
        returned object.
        If found, the cost of the compact assignment is the same as in the
        original assignment and it preserves the values of 'active' variables.
        Returns nullptr if a compact assignment was not found.
        This method only works in homogenous mode, and it only swaps equivalent
        vehicles (vehicles with the same start and end nodes). When creating the
        compact assignment, the empty plan is replaced by the route assigned to
        the compatible vehicle with the highest id. Note that with more complex
        constraints on vehicle variables, this method might fail even if a compact
        solution exists.
        This method changes the vehicle and dimension variables as necessary.
        While compacting the solution, only basic checks on vehicle variables are
        performed; if one of these checks fails no attempts to repair it are made
        (instead, the method returns nullptr).
        """
        return _pywrapcp.RoutingModel_CompactAssignment(self, assignment)

    def CompactAndCheckAssignment(self, assignment):
        """
        Same as CompactAssignment() but also checks the validity of the final
        compact solution; if it is not valid, no attempts to repair it are made
        (instead, the method returns nullptr).
        """
        return _pywrapcp.RoutingModel_CompactAndCheckAssignment(self, assignment)

    def AddToAssignment(self, var):
        """ Adds an extra variable to the vehicle routing assignment."""
        return _pywrapcp.RoutingModel_AddToAssignment(self, var)

    def AddIntervalToAssignment(self, interval):
        return _pywrapcp.RoutingModel_AddIntervalToAssignment(self, interval)

    def PackCumulsOfOptimizerDimensionsFromAssignment(self, original_assignment, duration_limit, time_limit_was_reached=None):
        """
        For every dimension in the model with an optimizer in
        local/global_dimension_optimizers_, this method tries to pack the cumul
        values of the dimension, such that:
        - The cumul costs (span costs, soft lower and upper bound costs, etc) are
          minimized.
        - The cumuls of the ends of the routes are minimized for this given
          minimal cumul cost.
        - Given these minimal end cumuls, the route start cumuls are maximized.
        Returns the assignment resulting from allocating these packed cumuls with
        the solver, and nullptr if these cumuls could not be set by the solver.
        """
        return _pywrapcp.RoutingModel_PackCumulsOfOptimizerDimensionsFromAssignment(self, original_assignment, duration_limit, time_limit_was_reached)

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

    def AddLocalSearchFilter(self, filter):
        """
        Adds a custom local search filter to the list of filters used to speed up
        local search by pruning unfeasible variable assignments.
        Calling this method after the routing model has been closed (CloseModel()
        or Solve() has been called) has no effect.
        The routing model does not take ownership of the filter.
        """
        return _pywrapcp.RoutingModel_AddLocalSearchFilter(self, filter)

    def Start(self, vehicle):
        """
        Model inspection.
        Returns the variable index of the starting node of a vehicle route.
        """
        return _pywrapcp.RoutingModel_Start(self, vehicle)

    def End(self, vehicle):
        """ Returns the variable index of the ending node of a vehicle route."""
        return _pywrapcp.RoutingModel_End(self, vehicle)

    def IsStart(self, index):
        """ Returns true if 'index' represents the first node of a route."""
        return _pywrapcp.RoutingModel_IsStart(self, index)

    def IsEnd(self, index):
        """ Returns true if 'index' represents the last node of a route."""
        return _pywrapcp.RoutingModel_IsEnd(self, index)

    def VehicleIndex(self, index):
        """
        Returns the vehicle of the given start/end index, and -1 if the given
        index is not a vehicle start/end.
        """
        return _pywrapcp.RoutingModel_VehicleIndex(self, index)

    def Next(self, assignment, index):
        """
        Assignment inspection
        Returns the variable index of the node directly after the node
        corresponding to 'index' in 'assignment'.
        """
        return _pywrapcp.RoutingModel_Next(self, assignment, index)

    def IsVehicleUsed(self, assignment, vehicle):
        """ Returns true if the route of 'vehicle' is non empty in 'assignment'."""
        return _pywrapcp.RoutingModel_IsVehicleUsed(self, assignment, vehicle)

    def NextVar(self, index):
        """
        Returns the next variable of the node corresponding to index. Note that
        NextVar(index) == index is equivalent to ActiveVar(index) == 0.
        """
        return _pywrapcp.RoutingModel_NextVar(self, index)

    def ActiveVar(self, index):
        """ Returns the active variable of the node corresponding to index."""
        return _pywrapcp.RoutingModel_ActiveVar(self, index)

    def ActiveVehicleVar(self, vehicle):
        """
        Returns the active variable of the vehicle. It will be equal to 1 iff the
        route of the vehicle is not empty, 0 otherwise.
        """
        return _pywrapcp.RoutingModel_ActiveVehicleVar(self, vehicle)

    def VehicleRouteConsideredVar(self, vehicle):
        """
        Returns the variable specifying whether or not the given vehicle route is
        considered for costs and constraints. It will be equal to 1 iff the route
        of the vehicle is not empty OR vehicle_used_when_empty_[vehicle] is true.
        """
        return _pywrapcp.RoutingModel_VehicleRouteConsideredVar(self, vehicle)

    def VehicleVar(self, index):
        """
        Returns the vehicle variable of the node corresponding to index. Note that
        VehicleVar(index) == -1 is equivalent to ActiveVar(index) == 0.
        """
        return _pywrapcp.RoutingModel_VehicleVar(self, index)

    def ResourceVar(self, vehicle, resource_group):
        """
        Returns the resource variable for the given vehicle index in the given
        resource group. If a vehicle doesn't require a resource from the
        corresponding resource group, then ResourceVar(v, r_g) == -1.
        """
        return _pywrapcp.RoutingModel_ResourceVar(self, vehicle, resource_group)

    def CostVar(self):
        """ Returns the global cost variable which is being minimized."""
        return _pywrapcp.RoutingModel_CostVar(self)

    def GetArcCostForVehicle(self, from_index, to_index, vehicle):
        """
        Returns the cost of the transit arc between two nodes for a given vehicle.
        Input are variable indices of node. This returns 0 if vehicle < 0.
        """
        return _pywrapcp.RoutingModel_GetArcCostForVehicle(self, from_index, to_index, vehicle)

    def CostsAreHomogeneousAcrossVehicles(self):
        """ Whether costs are homogeneous across all vehicles."""
        return _pywrapcp.RoutingModel_CostsAreHomogeneousAcrossVehicles(self)

    def GetHomogeneousCost(self, from_index, to_index):
        """
        Returns the cost of the segment between two nodes supposing all vehicle
        costs are the same (returns the cost for the first vehicle otherwise).
        """
        return _pywrapcp.RoutingModel_GetHomogeneousCost(self, from_index, to_index)

    def GetArcCostForFirstSolution(self, from_index, to_index):
        """
        Returns the cost of the arc in the context of the first solution strategy.
        This is typically a simplification of the actual cost; see the .cc.
        """
        return _pywrapcp.RoutingModel_GetArcCostForFirstSolution(self, from_index, to_index)

    def GetArcCostForClass(self, from_index, to_index, cost_class_index):
        """
        Returns the cost of the segment between two nodes for a given cost
        class. Input are variable indices of nodes and the cost class.
        Unlike GetArcCostForVehicle(), if cost_class is kNoCost, then the
        returned cost won't necessarily be zero: only some of the components
        of the cost that depend on the cost class will be omited. See the code
        for details.
        """
        return _pywrapcp.RoutingModel_GetArcCostForClass(self, from_index, to_index, cost_class_index)

    def GetCostClassIndexOfVehicle(self, vehicle):
        """ Get the cost class index of the given vehicle."""
        return _pywrapcp.RoutingModel_GetCostClassIndexOfVehicle(self, vehicle)

    def HasVehicleWithCostClassIndex(self, cost_class_index):
        """
        Returns true iff the model contains a vehicle with the given
        cost_class_index.
        """
        return _pywrapcp.RoutingModel_HasVehicleWithCostClassIndex(self, cost_class_index)

    def GetCostClassesCount(self):
        """ Returns the number of different cost classes in the model."""
        return _pywrapcp.RoutingModel_GetCostClassesCount(self)

    def GetNonZeroCostClassesCount(self):
        """ Ditto, minus the 'always zero', built-in cost class."""
        return _pywrapcp.RoutingModel_GetNonZeroCostClassesCount(self)

    def GetVehicleClassIndexOfVehicle(self, vehicle):
        return _pywrapcp.RoutingModel_GetVehicleClassIndexOfVehicle(self, vehicle)

    def GetVehicleOfClass(self, vehicle_class):
        """
        Returns a vehicle of the given vehicle class, and -1 if there are no
        vehicles for this class.
        """
        return _pywrapcp.RoutingModel_GetVehicleOfClass(self, vehicle_class)

    def GetVehicleClassesCount(self):
        """ Returns the number of different vehicle classes in the model."""
        return _pywrapcp.RoutingModel_GetVehicleClassesCount(self)

    def GetSameVehicleIndicesOfIndex(self, node):
        """ Returns variable indices of nodes constrained to be on the same route."""
        return _pywrapcp.RoutingModel_GetSameVehicleIndicesOfIndex(self, node)

    def GetVehicleTypeContainer(self):
        return _pywrapcp.RoutingModel_GetVehicleTypeContainer(self)

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

    def DebugOutputAssignment(self, solution_assignment, dimension_to_print):
        """
        Print some debugging information about an assignment, including the
        feasible intervals of the CumulVar for dimension "dimension_to_print"
        at each step of the routes.
        If "dimension_to_print" is omitted, all dimensions will be printed.
        """
        return _pywrapcp.RoutingModel_DebugOutputAssignment(self, solution_assignment, dimension_to_print)

    def CheckIfAssignmentIsFeasible(self, assignment, call_at_solution_monitors):
        """
        Returns a vector cumul_bounds, for which cumul_bounds[i][j] is a pair
        containing the minimum and maximum of the CumulVar of the jth node on
        route i.
        - cumul_bounds[i][j].first is the minimum.
        - cumul_bounds[i][j].second is the maximum.
        Checks if an assignment is feasible.
        """
        return _pywrapcp.RoutingModel_CheckIfAssignmentIsFeasible(self, assignment, call_at_solution_monitors)

    def solver(self):
        """
        Returns the underlying constraint solver. Can be used to add extra
        constraints and/or modify search algorithms.
        """
        return _pywrapcp.RoutingModel_solver(self)

    def CheckLimit(self, *args):
        """
        Returns true if the search limit has been crossed with the given time
        offset.
        """
        return _pywrapcp.RoutingModel_CheckLimit(self, *args)

    def RemainingTime(self):
        """ Returns the time left in the search limit."""
        return _pywrapcp.RoutingModel_RemainingTime(self)

    def UpdateTimeLimit(self, time_limit):
        """ Updates the time limit of the search limit."""
        return _pywrapcp.RoutingModel_UpdateTimeLimit(self, time_limit)

    def TimeBuffer(self):
        """ Returns the time buffer to safely return a solution."""
        return _pywrapcp.RoutingModel_TimeBuffer(self)

    def GetMutableCPSatInterrupt(self):
        """ Returns the atomic<bool> to stop the CP-SAT solver."""
        return _pywrapcp.RoutingModel_GetMutableCPSatInterrupt(self)

    def GetMutableCPInterrupt(self):
        """ Returns the atomic<bool> to stop the CP solver."""
        return _pywrapcp.RoutingModel_GetMutableCPInterrupt(self)

    def CancelSearch(self):
        """ Cancels the current search."""
        return _pywrapcp.RoutingModel_CancelSearch(self)

    def nodes(self):
        """
        Sizes and indices
        Returns the number of nodes in the model.
        """
        return _pywrapcp.RoutingModel_nodes(self)

    def vehicles(self):
        """ Returns the number of vehicle routes in the model."""
        return _pywrapcp.RoutingModel_vehicles(self)

    def Size(self):
        """ Returns the number of next variables in the model."""
        return _pywrapcp.RoutingModel_Size(self)

    def GetNumberOfDecisionsInFirstSolution(self, search_parameters):
        """
        Returns statistics on first solution search, number of decisions sent to
        filters, number of decisions rejected by filters.
        """
        return _pywrapcp.RoutingModel_GetNumberOfDecisionsInFirstSolution(self, search_parameters)

    def GetNumberOfRejectsInFirstSolution(self, search_parameters):
        return _pywrapcp.RoutingModel_GetNumberOfRejectsInFirstSolution(self, search_parameters)

    def GetAutomaticFirstSolutionStrategy(self):
        """ Returns the automatic first solution strategy selected."""
        return _pywrapcp.RoutingModel_GetAutomaticFirstSolutionStrategy(self)

    def IsMatchingModel(self):
        """ Returns true if a vehicle/node matching problem is detected."""
        return _pywrapcp.RoutingModel_IsMatchingModel(self)

    def AreRoutesInterdependent(self, parameters):
        """
        Returns true if routes are interdependent. This means that any
        modification to a route might impact another.
        """
        return _pywrapcp.RoutingModel_AreRoutesInterdependent(self, parameters)

    def MakeGuidedSlackFinalizer(self, dimension, initializer):
        """
        The next few members are in the public section only for testing purposes.

        MakeGuidedSlackFinalizer creates a DecisionBuilder for the slacks of a
        dimension using a callback to choose which values to start with.
        The finalizer works only when all next variables in the model have
        been fixed. It has the following two characteristics:
        1. It follows the routes defined by the nexts variables when choosing a
           variable to make a decision on.
        2. When it comes to choose a value for the slack of node i, the decision
           builder first calls the callback with argument i, and supposingly the
           returned value is x it creates decisions slack[i] = x, slack[i] = x +
           1, slack[i] = x - 1, slack[i] = x + 2, etc.
        """
        return _pywrapcp.RoutingModel_MakeGuidedSlackFinalizer(self, dimension, initializer)

    def MakeSelfDependentDimensionFinalizer(self, dimension):
        """
        MakeSelfDependentDimensionFinalizer is a finalizer for the slacks of a
        self-dependent dimension. It makes an extensive use of the caches of the
        state dependent transits.
        In detail, MakeSelfDependentDimensionFinalizer returns a composition of a
        local search decision builder with a greedy descent operator for the cumul
        of the start of each route and a guided slack finalizer. Provided there
        are no time windows and the maximum slacks are large enough, once the
        cumul of the start of route is fixed, the guided finalizer can find
        optimal values of the slacks for the rest of the route in time
        proportional to the length of the route. Therefore the composed finalizer
        generally works in time O(log(t)*n*m), where t is the latest possible
        departute time, n is the number of nodes in the network and m is the
        number of vehicles.
        """
        return _pywrapcp.RoutingModel_MakeSelfDependentDimensionFinalizer(self, dimension)

    def GetPathsMetadata(self):
        return _pywrapcp.RoutingModel_GetPathsMetadata(self)