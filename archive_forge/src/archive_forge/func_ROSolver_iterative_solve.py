from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np
def ROSolver_iterative_solve(model_data, config):
    """
    GRCS algorithm implementation
    :model_data: ROSolveData object with deterministic model information
    :config: ConfigBlock for the instance being solved
    """
    violation = list((p for p in config.nominal_uncertain_param_vals))
    constraints = [c for c in model_data.working_model.component_data_objects(Constraint) if c.equality and c not in ComponentSet(model_data.working_model.util.decision_rule_eqns)]
    model_data.working_model.util.h_x_q_constraints = ComponentSet()
    for c in constraints:
        coeff_matching_success, robust_infeasible = coefficient_matching(model=model_data.working_model, constraint=c, uncertain_params=model_data.working_model.util.uncertain_params, config=config)
        if not coeff_matching_success and (not robust_infeasible):
            config.progress_logger.error(f'Equality constraint {c.name!r} cannot be guaranteed to be robustly feasible, given the current partitioning among first-stage, second-stage, and state variables. Consider editing this constraint to reference some second-stage and/or state variable(s).')
            raise ValueError('Coefficient matching unsuccessful. See the solver logs.')
        elif not coeff_matching_success and robust_infeasible:
            config.progress_logger.info(f'PyROS has determined that the model is robust infeasible. One reason for this is that the equality constraint {c.name} cannot be satisfied against all realizations of uncertainty, given the current partitioning between first-stage, second-stage, and state variables. Consider editing this constraint to reference some (additional) second-stage and/or state variable(s).')
            return (None, None)
        else:
            pass
    for c in model_data.working_model.util.h_x_q_constraints:
        c.deactivate()
    master_data = master_problem_methods.initial_construct_master(model_data)
    if config.p_robustness:
        master_data.master_model.p_robust_constraints = ConstraintList()
    master_data.master_model.scenarios[0, 0].transfer_attributes_from(master_data.original.clone())
    if len(master_data.master_model.scenarios[0, 0].util.uncertain_params) != len(violation):
        raise ValueError
    for i, v in enumerate(violation):
        master_data.master_model.scenarios[0, 0].util.uncertain_params[i].value = v
    if config.objective_focus is ObjectiveType.nominal:
        master_data.master_model.obj = Objective(expr=master_data.master_model.scenarios[0, 0].first_stage_objective + master_data.master_model.scenarios[0, 0].second_stage_objective)
    elif config.objective_focus is ObjectiveType.worst_case:
        master_data.master_model.zeta = Var(initialize=value(master_data.master_model.scenarios[0, 0].first_stage_objective + master_data.master_model.scenarios[0, 0].second_stage_objective, exception=False))
        master_data.master_model.obj = Objective(expr=master_data.master_model.zeta)
        master_data.master_model.scenarios[0, 0].epigraph_constr = Constraint(expr=master_data.master_model.scenarios[0, 0].first_stage_objective + master_data.master_model.scenarios[0, 0].second_stage_objective <= master_data.master_model.zeta)
        master_data.master_model.scenarios[0, 0].util.first_stage_variables.append(master_data.master_model.zeta)
    master_data.original.util.deterministic_constraints = ComponentSet((c for c in master_data.original.component_data_objects(Constraint, descend_into=True)))
    separation_model = separation_problem_methods.make_separation_problem(model_data=master_data, config=config)
    evaluate_and_log_component_stats(model_data=model_data, separation_model=separation_model, config=config)
    separation_data = SeparationProblemData()
    separation_data.separation_model = separation_model
    separation_data.points_separated = []
    separation_data.points_added_to_master = [config.nominal_uncertain_param_vals]
    separation_data.constraint_violations = []
    separation_data.total_global_separation_solves = 0
    separation_data.timing = master_data.timing
    separation_data.separation_problem_subsolver_statuses = []
    if config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS:
        separation_data.idxs_of_master_scenarios = [config.uncertainty_set.scenarios.index(tuple(config.nominal_uncertain_param_vals))]
    else:
        separation_data.idxs_of_master_scenarios = None
    nominal_data = Block()
    nominal_data.nom_fsv_vals = []
    nominal_data.nom_ssv_vals = []
    nominal_data.nom_first_stage_cost = 0
    nominal_data.nom_second_stage_cost = 0
    nominal_data.nom_obj = 0
    timing_data = Block()
    timing_data.total_master_solve_time = 0
    timing_data.total_separation_local_time = 0
    timing_data.total_separation_global_time = 0
    timing_data.total_dr_polish_time = 0
    dr_var_lists_original = []
    dr_var_lists_polished = []
    master_dr_var_set = ComponentSet(chain(*tuple((indexed_var.values() for indexed_var in master_data.master_model.scenarios[0, 0].util.decision_rule_vars))))
    master_fsv_set = ComponentSet((var for var in master_data.master_model.scenarios[0, 0].util.first_stage_variables if var not in master_dr_var_set))
    master_nom_ssv_set = ComponentSet(master_data.master_model.scenarios[0, 0].util.second_stage_variables)
    previous_master_fsv_vals = ComponentMap(((var, None) for var in master_fsv_set))
    previous_master_dr_var_vals = ComponentMap(((var, None) for var in master_dr_var_set))
    previous_master_nom_ssv_vals = ComponentMap(((var, None) for var in master_nom_ssv_set))
    first_iter_master_fsv_vals = ComponentMap(((var, None) for var in master_fsv_set))
    first_iter_master_nom_ssv_vals = ComponentMap(((var, None) for var in master_nom_ssv_set))
    first_iter_dr_var_vals = ComponentMap(((var, None) for var in master_dr_var_set))
    nom_master_util_blk = master_data.master_model.scenarios[0, 0].util
    dr_var_scaled_expr_map = get_dr_var_to_scaled_expr_map(decision_rule_vars=nom_master_util_blk.decision_rule_vars, decision_rule_eqns=nom_master_util_blk.decision_rule_eqns, second_stage_vars=nom_master_util_blk.second_stage_variables, uncertain_params=nom_master_util_blk.uncertain_params)
    dr_var_to_ssv_map = ComponentMap()
    dr_ssv_zip = zip(nom_master_util_blk.decision_rule_vars, nom_master_util_blk.second_stage_variables)
    for indexed_dr_var, ssv in dr_ssv_zip:
        for drvar in indexed_dr_var.values():
            dr_var_to_ssv_map[drvar] = ssv
    IterationLogRecord.log_header(config.progress_logger.info)
    k = 0
    master_statuses = []
    while config.max_iter == -1 or k < config.max_iter:
        master_data.iteration = k
        if k > 0 and config.p_robustness:
            master_problem_methods.add_p_robust_constraint(model_data=master_data, config=config)
        config.progress_logger.debug(f'PyROS working on iteration {k}...')
        master_soln = master_problem_methods.solve_master(model_data=master_data, config=config)
        timing_data.total_master_solve_time += get_time_from_solver(master_soln.results)
        if k > 0:
            timing_data.total_master_solve_time += get_time_from_solver(master_soln.feasibility_problem_results)
        master_statuses.append(master_soln.results.solver.termination_condition)
        master_soln.master_problem_subsolver_statuses = master_statuses
        if master_soln.master_subsolver_results[1] is pyrosTerminationCondition.robust_infeasible:
            term_cond = pyrosTerminationCondition.robust_infeasible
        elif master_soln.pyros_termination_condition is pyrosTerminationCondition.subsolver_error:
            term_cond = pyrosTerminationCondition.subsolver_error
        elif master_soln.pyros_termination_condition is pyrosTerminationCondition.time_out:
            term_cond = pyrosTerminationCondition.time_out
        else:
            term_cond = None
        if term_cond in {pyrosTerminationCondition.subsolver_error, pyrosTerminationCondition.time_out, pyrosTerminationCondition.robust_infeasible}:
            log_record = IterationLogRecord(iteration=k, objective=None, first_stage_var_shift=None, second_stage_var_shift=None, dr_var_shift=None, num_violated_cons=None, max_violation=None, dr_polishing_success=None, all_sep_problems_solved=None, global_separation=None, elapsed_time=get_main_elapsed_time(model_data.timing))
            log_record.log(config.progress_logger.info)
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=term_cond, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
            return (model_data, [])
        if k == 0:
            for val in master_soln.fsv_vals:
                nominal_data.nom_fsv_vals.append(val)
            for val in master_soln.ssv_vals:
                nominal_data.nom_ssv_vals.append(val)
            nominal_data.nom_first_stage_cost = master_soln.first_stage_objective
            nominal_data.nom_second_stage_cost = master_soln.second_stage_objective
            nominal_data.nom_obj = value(master_data.master_model.obj)
        polishing_successful = True
        if config.decision_rule_order != 0 and len(config.second_stage_variables) > 0 and (k != 0):
            for varslist in master_data.master_model.scenarios[0, 0].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_original.append(vals)
            polishing_results, polishing_successful = master_problem_methods.minimize_dr_vars(model_data=master_data, config=config)
            timing_data.total_dr_polish_time += get_time_from_solver(polishing_results)
            for varslist in master_data.master_model.scenarios[0, 0].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_polished.append(vals)
        current_master_fsv_vals = ComponentMap(((var, value(var)) for var in master_fsv_set))
        current_master_nom_ssv_vals = ComponentMap(((var, value(var)) for var in master_nom_ssv_set))
        current_master_dr_var_vals = ComponentMap(((var, value(expr)) for var, expr in dr_var_scaled_expr_map.items()))
        if k > 0:
            first_stage_var_shift = evaluate_first_stage_var_shift(current_master_fsv_vals=current_master_fsv_vals, previous_master_fsv_vals=previous_master_fsv_vals, first_iter_master_fsv_vals=first_iter_master_fsv_vals)
            second_stage_var_shift = evaluate_second_stage_var_shift(current_master_nom_ssv_vals=current_master_nom_ssv_vals, previous_master_nom_ssv_vals=previous_master_nom_ssv_vals, first_iter_master_nom_ssv_vals=first_iter_master_nom_ssv_vals)
            dr_var_shift = evaluate_dr_var_shift(current_master_dr_var_vals=current_master_dr_var_vals, previous_master_dr_var_vals=previous_master_dr_var_vals, first_iter_master_nom_ssv_vals=first_iter_master_nom_ssv_vals, dr_var_to_ssv_map=dr_var_to_ssv_map)
        else:
            for fsv in first_iter_master_fsv_vals:
                first_iter_master_fsv_vals[fsv] = value(fsv)
            for ssv in first_iter_master_nom_ssv_vals:
                first_iter_master_nom_ssv_vals[ssv] = value(ssv)
            for drvar in first_iter_dr_var_vals:
                first_iter_dr_var_vals[drvar] = value(dr_var_scaled_expr_map[drvar])
            first_stage_var_shift = None
            second_stage_var_shift = None
            dr_var_shift = None
        if config.time_limit:
            elapsed = get_main_elapsed_time(model_data.timing)
            if elapsed >= config.time_limit:
                iter_log_record = IterationLogRecord(iteration=k, objective=value(master_data.master_model.obj), first_stage_var_shift=first_stage_var_shift, second_stage_var_shift=second_stage_var_shift, dr_var_shift=dr_var_shift, num_violated_cons=None, max_violation=None, dr_polishing_success=polishing_successful, all_sep_problems_solved=None, global_separation=None, elapsed_time=elapsed)
                update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=pyrosTerminationCondition.time_out, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
                iter_log_record.log(config.progress_logger.info)
                return (model_data, [])
        separation_data.opt_fsv_vals = [v.value for v in master_soln.master_model.scenarios[0, 0].util.first_stage_variables]
        separation_data.opt_ssv_vals = master_soln.ssv_vals
        separation_data.master_scenarios = master_data.master_model.scenarios
        if config.objective_focus is ObjectiveType.worst_case:
            separation_model.util.zeta = value(master_soln.master_model.obj)
        separation_data.iteration = k
        separation_data.master_nominal_scenario = master_data.master_model.scenarios[0, 0]
        separation_data.master_model = master_data.master_model
        separation_results = separation_problem_methods.solve_separation_problem(model_data=separation_data, config=config)
        separation_data.separation_problem_subsolver_statuses.extend([res.solver.termination_condition for res in separation_results.generate_subsolver_results()])
        if separation_results.solved_globally:
            separation_data.total_global_separation_solves += 1
        timing_data.total_separation_local_time += separation_results.evaluate_local_solve_time(get_time_from_solver)
        timing_data.total_separation_global_time += separation_results.evaluate_global_solve_time(get_time_from_solver)
        if separation_results.found_violation:
            scaled_violations = separation_results.scaled_violations
            if scaled_violations is not None:
                separation_data.constraint_violations.append(scaled_violations.values())
        separation_data.points_separated = separation_results.violating_param_realization
        scaled_violations = [solve_call_res.scaled_violations[con] for con, solve_call_res in separation_results.main_loop_results.solver_call_results.items() if solve_call_res.scaled_violations is not None]
        if scaled_violations:
            max_sep_con_violation = max(scaled_violations)
        else:
            max_sep_con_violation = None
        num_violated_cons = len(separation_results.violated_performance_constraints)
        all_sep_problems_solved = len(scaled_violations) == len(separation_model.util.performance_constraints) and (not separation_results.subsolver_error) and (not separation_results.time_out)
        iter_log_record = IterationLogRecord(iteration=k, objective=value(master_data.master_model.obj), first_stage_var_shift=first_stage_var_shift, second_stage_var_shift=second_stage_var_shift, dr_var_shift=dr_var_shift, num_violated_cons=num_violated_cons, max_violation=max_sep_con_violation, dr_polishing_success=polishing_successful, all_sep_problems_solved=all_sep_problems_solved, global_separation=separation_results.solved_globally, elapsed_time=get_main_elapsed_time(model_data.timing))
        elapsed = get_main_elapsed_time(model_data.timing)
        if separation_results.time_out:
            termination_condition = pyrosTerminationCondition.time_out
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
            iter_log_record.log(config.progress_logger.info)
            return (model_data, separation_results)
        if separation_results.subsolver_error:
            termination_condition = pyrosTerminationCondition.subsolver_error
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
            iter_log_record.log(config.progress_logger.info)
            return (model_data, separation_results)
        robustness_certified = separation_results.robustness_certified
        if robustness_certified:
            if config.bypass_global_separation:
                config.progress_logger.warning('Option to bypass global separation was chosen. Robust feasibility and optimality of the reported solution are not guaranteed.')
            robust_optimal = config.solve_master_globally and config.objective_focus is ObjectiveType.worst_case
            if robust_optimal:
                termination_condition = pyrosTerminationCondition.robust_optimal
            else:
                termination_condition = pyrosTerminationCondition.robust_feasible
            update_grcs_solve_data(pyros_soln=model_data, k=k, term_cond=termination_condition, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
            iter_log_record.log(config.progress_logger.info)
            return (model_data, separation_results)
        master_problem_methods.add_scenario_to_master(model_data=master_data, violations=separation_results.violating_param_realization)
        separation_data.points_added_to_master.append(separation_results.violating_param_realization)
        config.progress_logger.debug('Points added to master:')
        config.progress_logger.debug(np.array([pt for pt in separation_data.points_added_to_master]))
        for var, val in separation_results.violating_separation_variable_values.items():
            master_var = master_data.master_model.scenarios[k + 1, 0].find_component(var)
            master_var.set_value(val)
        k += 1
        iter_log_record.log(config.progress_logger.info)
        previous_master_fsv_vals = current_master_fsv_vals
        previous_master_nom_ssv_vals = current_master_nom_ssv_vals
        previous_master_dr_var_vals = current_master_dr_var_vals
    update_grcs_solve_data(pyros_soln=model_data, k=k - 1, term_cond=pyrosTerminationCondition.max_iter, nominal_data=nominal_data, timing_data=timing_data, separation_data=separation_data, master_soln=master_soln)
    return (model_data, separation_results)