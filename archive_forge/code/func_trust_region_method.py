import logging
from pyomo.core.base.range import NumericRange
from pyomo.common.config import (
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.opt import SolverFactory
def trust_region_method(model, decision_variables, ext_fcn_surrogate_map_rule, config):
    """
    The main driver of the Trust Region algorithm method.

    Parameters
    ----------
    model : ConcreteModel
        The user's model to be solved.
    degrees_of_freedom_variables : List of Vars
        User-supplied input. The user must provide a list of vars which
        are the degrees of freedom or decision variables within
        the model.
    ext_fcn_surrogate_map_rule : Function, optional
        In the 2020 Yoshio/Biegler paper, this is referred to as
        the basis function `b(w)`.
        This is the low-fidelity model with which to solve the original
        process model problem and which is integrated into the
        surrogate model.
        The default is 0 (i.e., no basis function rule.)
    config : ConfigDict
        This holds the solver and TRF-specific configuration options.

    """
    TRFLogger = IterationLogger()
    TRFilter = Filter()
    interface = TRFInterface(model, decision_variables, ext_fcn_surrogate_map_rule, config)
    rebuildSM = False
    obj_val, feasibility = interface.initializeProblem()
    feasibility_k = feasibility
    obj_val_k = obj_val
    step_norm_k = 0
    trust_radius = config.trust_radius
    iteration = 0
    TRFLogger.newIteration(iteration, feasibility_k, obj_val_k, trust_radius, step_norm_k)
    TRFLogger.logIteration()
    if config.verbose:
        TRFLogger.printIteration()
    while iteration < config.maximum_iterations:
        iteration += 1
        if feasibility_k <= config.feasibility_termination and step_norm_k <= config.step_size_termination:
            print('EXIT: Optimal solution found.')
            interface.model.display()
            break
        if trust_radius <= config.minimum_radius and abs(feasibility_k - feasibility) < config.feasibility_termination:
            if subopt_flag:
                logger.warning('WARNING: Insufficient progress.')
                print('EXIT: Feasible solution found.')
                break
            else:
                subopt_flag = True
        else:
            subopt_flag = False
        interface.updateDecisionVariableBounds(trust_radius)
        if rebuildSM:
            interface.updateSurrogateModel()
        obj_val_k, step_norm_k, feasibility_k = interface.solveModel()
        TRFLogger.newIteration(iteration, feasibility_k, obj_val_k, trust_radius, step_norm_k)
        filterElement = FilterElement(obj_val_k, feasibility_k)
        if not TRFilter.isAcceptable(filterElement, config.maximum_feasibility):
            TRFLogger.iterrecord.rejected = True
            trust_radius = max(config.minimum_radius, step_norm_k * config.radius_update_param_gamma_c)
            rebuildSM = False
            interface.rejectStep()
            TRFLogger.logIteration()
            if config.verbose:
                TRFLogger.printIteration()
            continue
        if obj_val - obj_val_k >= config.switch_condition_kappa_theta * pow(feasibility, config.switch_condition_gamma_s) and feasibility <= config.minimum_feasibility:
            TRFLogger.iterrecord.fStep = True
            trust_radius = min(max(step_norm_k * config.radius_update_param_gamma_e, trust_radius), config.maximum_radius)
        else:
            TRFLogger.iterrecord.thetaStep = True
            filterElement = FilterElement(obj_val_k - config.param_filter_gamma_f * feasibility_k, (1 - config.param_filter_gamma_theta) * feasibility_k)
            TRFilter.addToFilter(filterElement)
            rho_k = (feasibility - feasibility_k + config.feasibility_termination) / max(feasibility, config.feasibility_termination)
            if rho_k < config.ratio_test_param_eta_1 or feasibility > config.minimum_feasibility:
                trust_radius = max(config.minimum_radius, config.radius_update_param_gamma_c * step_norm_k)
            elif rho_k >= config.ratio_test_param_eta_2:
                trust_radius = min(config.maximum_radius, max(trust_radius, config.radius_update_param_gamma_e * step_norm_k))
        TRFLogger.updateIteration(trustRadius=trust_radius)
        rebuildSM = True
        feasibility = feasibility_k
        obj_val = obj_val_k
        TRFLogger.logIteration()
        if config.verbose:
            TRFLogger.printIteration()
    if iteration >= config.maximum_iterations:
        logger.warning('EXIT: Maximum iterations reached: {}.'.format(config.maximum_iterations))
    return interface.model