from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
@TransformationFactory.register('gdp.cuttingplane', doc='Relaxes a linear disjunctive model by adding cuts from convex hull to Big-M reformulation.')
class CuttingPlane_Transformation(Transformation):
    """Relax convex disjunctive model by forming the bigm relaxation and then
    iteratively adding cuts from the hull relaxation (or the hull relaxation
    after some basic steps) in order to strengthen the formulation.

    Note that gdp.cuttingplane is not a structural transformation: If variables
    on the model are fixed, they will be treated as data, and unfixing them
    after transformation will very likely result in an invalid model.

    This transformation accepts the following keyword arguments:

    Parameters
    ----------
    solver : Solver name (as string) to use to solve relaxed BigM and separation
             problems
    solver_options : dictionary of options to pass to the solver
    stream_solver : Whether or not to display solver output
    verbose : Enable verbose output from cuttingplanes algorithm
    cuts_name : Optional name for the IndexedConstraint containing the projected
                cuts (must be a unique name with respect to the instance)
    minimum_improvement_threshold : Stopping criterion based on improvement in
                                    Big-M relaxation. This is the minimum
                                    difference in relaxed BigM objective
                                    values between consecutive iterations
    separation_objective_threshold : Stopping criterion based on separation
                                     objective. If separation objective is not
                                     at least this large, cut generation will
                                     terminate.
    cut_filtering_threshold : Stopping criterion based on effectiveness of the
                              generated cut: This is the amount by which
                              a cut must be violated at the relaxed bigM
                              solution in order to be added to the bigM model
    max_number_of_cuts : The maximum number of cuts to add to the big-M model
    norm : norm to use in the objective of the separation problem
    tighten_relaxation : callback to modify the GDP model before the hull
                         relaxation is taken (e.g. could be used to perform
                         basic steps)
    create_cuts : callback to create cuts using the solved relaxed bigM and hull
                  problems
    post_process_cut : callback to perform post-processing on created cuts
    back_off_problem_tolerance : tolerance to use while post-processing
    zero_tolerance : Tolerance at which a float will be considered 0 when
                     using Fourier-Motzkin elimination to create cuts.
    do_integer_arithmetic : Whether or not to require Fourier-Motzkin elimination
                            to do integer arithmetic. Only possible when all
                            data is integer.
    tight_constraint_tolerance : Tolerance at which a constraint is considered
                                 tight for the Fourier-Motzkin cut generation
                                 procedure

    By default, the callbacks will be set such that the algorithm performed is
    as presented in [1], but with an additional post-processing procedure to
    reduce numerical error, which calculates the maximum violation of the cut
    subject to the relaxed hull constraints, and then pads the constraint by
    this violation plus an additional user-specified tolerance.

    In addition, the create_cuts_fme function provides an (exponential time)
    method of generating cuts which reduces numerical error (and can eliminate
    it if all data is integer). It collects the hull constraints which are
    tight at the solution of the separation problem. It creates a cut in the
    extended space perpendicular to  a composite normal vector created by
    summing the directions normal to these constraints. It then performs
    fourier-motzkin elimination on the collection of constraints and the cut
    to project out the disaggregated variables. The resulting constraint which
    is most violated by the relaxed bigM solution is then returned.

    References
    ----------
        [1] Sawaya, N. W., Grossmann, I. E. (2005). A cutting plane method for
        solving linear generalized disjunctive programming problems. Computers
        and Chemical Engineering, 29, 1891-1913
    """
    CONFIG = ConfigBlock('gdp.cuttingplane')
    CONFIG.declare('solver', ConfigValue(default='ipopt', domain=str, description='Solver to use for relaxed BigM problem and the separation\n        problem', doc='\n        This specifies the solver which will be used to solve LP relaxation\n        of the BigM problem and the separation problem. Note that this solver\n        must be able to handle a quadratic objective because of the separation\n        problem.\n        '))
    CONFIG.declare('minimum_improvement_threshold', ConfigValue(default=0.01, domain=NonNegativeFloat, description='Threshold value for difference in relaxed bigM problem objectives used to decide when to stop adding cuts', doc='\n        If the difference between the objectives in two consecutive iterations is\n        less than this value, the algorithm terminates without adding the cut\n        generated in the last iteration.  \n        '))
    CONFIG.declare('separation_objective_threshold', ConfigValue(default=0.01, domain=NonNegativeFloat, description='Threshold value used to decide when to stop adding cuts: If separation problem objective is not at least this quantity, cut generation will terminate.', doc='\n        If the separation problem objective (distance between relaxed bigM \n        solution and its projection onto the relaxed hull feasible region)\n        does not exceed this threshold, the algorithm will terminate.\n        '))
    CONFIG.declare('max_number_of_cuts', ConfigValue(default=100, domain=PositiveInt, description='The maximum number of cuts to add before the algorithm terminates.', doc='\n        If the algorithm does not terminate due to another criterion first,\n        cut generation will stop after adding this many cuts.\n        '))
    CONFIG.declare('norm', ConfigValue(default=2, domain=In([2, float('inf')]), description="Norm to use in the separation problem: 2, or float('inf')", doc="\n        Norm used to calculate distance in the objective of the separation \n        problem which finds the nearest point on the hull relaxation region\n        to the current solution of the relaxed bigm problem.\n\n        Supported norms are the Euclidean norm (specify 2) and the infinity \n        norm (specify float('inf')). Note that the first makes the separation \n        problem objective quadratic and the latter makes it linear.\n        "))
    CONFIG.declare('verbose', ConfigValue(default=False, domain=bool, description='Flag to enable verbose output', doc='\n        If True, prints subproblem solutions, as well as potential and added cuts\n        during algorithm.\n\n        If False, only the relaxed BigM objective and minimal information about \n        cuts is logged.\n        '))
    CONFIG.declare('stream_solver', ConfigValue(default=False, domain=bool, description='If true, sets tee=True for every solve performed over\n        "the course of the algorithm'))
    CONFIG.declare('solver_options', ConfigBlock(implicit=True, description='Dictionary of solver options', doc='\n        Dictionary of solver options that will be set for the solver for both the\n        relaxed BigM and separation problem solves.\n        '))
    CONFIG.declare('tighten_relaxation', ConfigValue(default=do_not_tighten, description='Callback which takes the GDP formulation and returns a GDP formulation with a tighter hull relaxation', doc='\n        Function which accepts the GDP formulation of the problem and returns\n        a GDP formulation which the transformation will then take the hull\n        reformulation of.\n\n        Most typically, this callback would be used to apply basic steps before\n        taking the hull reformulation, but anything which tightens the GDP can \n        be performed here.\n        '))
    CONFIG.declare('create_cuts', ConfigValue(default=create_cuts_normal_vector, description='Callback which generates a list of cuts, given the solved relaxed bigM and relaxed hull solutions. If no cuts can be generated, returns None', doc='\n        Callback to generate cuts to be added to the bigM problem based on \n        solutions to the relaxed bigM problem and the separation problem.\n\n        Arguments\n        ---------\n        transBlock_rBigm: transformation block on relaxed bigM instance\n        transBlock_rHull: transformation block on relaxed hull instance\n        var_info: List of tuples (rBigM_var, rHull_var, xstar_param)\n        hull_to_bigm_map: For expression substitution, maps id(hull_var) to \n                          corresponding bigm var\n        rBigM_linear_constraints: list of linear constraints in relaxed bigM\n        rHull_vars: list of all variables in relaxed hull\n        disaggregated_vars: ComponentSet of disaggregated variables in hull \n                            reformulation\n        cut_threshold: Amount x* needs to be infeasible in generated cut in order\n                       to consider the cut for addition to the bigM model.\n        zero_tolerance: Tolerance at which a float will be treated as 0\n\n        Returns\n        -------\n        list of cuts to be added to bigM problem (and relaxed bigM problem),\n        represented as expressions using variables from the bigM model\n        '))
    CONFIG.declare('post_process_cut', ConfigValue(default=back_off_constraint_with_calculated_cut_violation, description='Callback which takes a generated cut and post processes it, presumably to back it off in the case of numerical error. Set to None if not post-processing is desired.', doc="\n        Callback to adjust a cut returned from create_cuts before adding it to\n        the model, presumably to make it more conservative in case of numerical\n        error.\n\n        Arguments\n        ---------\n        cut: the cut to be made more conservative, a Constraint\n        transBlock_rHull: the relaxed hull model's transformation Block.\n        bigm_to_hull_map: Dictionary mapping ids of bigM variables to the \n                          corresponding variables on the relaxed hull instance.\n        opt: SolverFactory object for subproblem solves in this procedure\n        stream_solver: Whether or not to set tee=True while solving.\n        TOL: A tolerance\n\n        Returns\n        -------\n        None, modifies the cut in place\n        "))
    CONFIG.declare('back_off_problem_tolerance', ConfigValue(default=1e-08, domain=NonNegativeFloat, description='Tolerance to pass to the post_process_cut callback.', doc="\n        Tolerance passed to the post_process_cut callback.\n\n        Depending on the callback, different values could make sense, but \n        something on the order of the solver's optimality or constraint \n        tolerances is appropriate.\n        "))
    CONFIG.declare('cut_filtering_threshold', ConfigValue(default=0.001, domain=NonNegativeFloat, description='Tolerance used to decide if a cut removes x* from the relaxed BigM problem by enough to be added to the bigM problem.', doc='\n        Absolute tolerance used to decide whether to keep a cut. We require\n        that, when evaluated at x* (the relaxed BigM optimal solution), the \n        cut be infeasible by at least this tolerance.\n        '))
    CONFIG.declare('zero_tolerance', ConfigValue(default=1e-09, domain=NonNegativeFloat, description='Tolerance at which floats are assumed to be 0 while performing Fourier-Motzkin elimination', doc='\n        Only relevant when create_cuts=create_cuts_fme, this sets the \n        zero_tolerance option for the Fourier-Motzkin elimination transformation.\n        '))
    CONFIG.declare('do_integer_arithmetic', ConfigValue(default=False, domain=bool, description='Only relevant if using Fourier-Motzkin Elimination (FME) and if all problem data is integer, requires FME transformation to perform integer arithmetic to eliminate numerical error.', doc='\n        Only relevant when create_cuts=create_cuts_fme and if all problem data \n        is integer, this sets the do_integer_arithmetic flag to true for the \n        FME transformation, meaning that the projection to the big-M space \n        can be done with exact precision.\n        '))
    CONFIG.declare('cuts_name', ConfigValue(default=None, domain=str, description='Optional name for the IndexedConstraint containing the projected cuts. Must be a unique name with respect to the instance.', doc='\n        Optional name for the IndexedConstraint containing the projected \n        constraints. If not specified, the cuts will be stored on a \n        private block created by the transformation, so if you want access \n        to them after the transformation, use this argument.\n\n        Must be a string which is a unique component name with respect to the \n        Block on which the transformation is called.\n        '))
    CONFIG.declare('tight_constraint_tolerance', ConfigValue(default=1e-06, domain=NonNegativeFloat, description='Tolerance at which a constraint is considered tight for the Fourier-Motzkin cut generation procedure.', doc='\n        For a constraint a^Tx <= b, the Fourier-Motzkin cut generation procedure\n        will consider the constraint tight (and add it to the set of constraints\n        being projected) when a^Tx - b is less than this tolerance. \n\n        It is recommended to set this tolerance to the constraint tolerance of\n        the solver being used.\n        '))

    def __init__(self):
        super(CuttingPlane_Transformation, self).__init__()

    def _apply_to(self, instance, bigM=None, **kwds):
        original_log_level = logger.level
        log_level = logger.getEffectiveLevel()
        try:
            self._config = self.CONFIG(kwds.pop('options', {}))
            self._config.set_value(kwds)
            if self._config.verbose and log_level > logging.INFO:
                logger.setLevel(logging.INFO)
                self.verbose = True
            elif log_level <= logging.INFO:
                self.verbose = True
            else:
                self.verbose = False
            instance_rBigM, cuts_obj, instance_rHull, var_info, transBlockName = self._setup_subproblems(instance, bigM, self._config.tighten_relaxation)
            self._generate_cuttingplanes(instance_rBigM, cuts_obj, instance_rHull, var_info, transBlockName)
            TransformationFactory('core.relax_integer_vars').apply_to(instance, undo=True)
        finally:
            del self._config
            del self.verbose
            logger.setLevel(original_log_level)

    def _setup_subproblems(self, instance, bigM, tighten_relaxation_callback):
        transBlockName, transBlock = self._add_transformation_block(instance)
        transBlock.all_vars = list((v for v in instance.component_data_objects(Var, descend_into=(Block, Disjunct), sort=SortComponents.deterministic) if not v.is_fixed()))
        nm = self._config.cuts_name
        if nm is None:
            cuts_obj = transBlock.cuts = Constraint(NonNegativeIntegers)
        else:
            if instance.component(nm) is not None:
                raise GDP_Error("cuts_name was specified as '%s', but this is already a component on the instance! Please specify a unique name." % nm)
            instance.add_component(nm, Constraint(NonNegativeIntegers))
            cuts_obj = instance.component(nm)
        bigMRelaxation = TransformationFactory('gdp.bigm')
        hullRelaxation = TransformationFactory('gdp.hull')
        relaxIntegrality = TransformationFactory('core.relax_integer_vars')
        tighter_instance = tighten_relaxation_callback(instance)
        instance_rHull = hullRelaxation.create_using(tighter_instance)
        relaxIntegrality.apply_to(instance_rHull, transform_deactivated_blocks=True)
        bigMRelaxation.apply_to(instance, bigM=bigM)
        relaxIntegrality.apply_to(instance, transform_deactivated_blocks=True)
        transBlock_rHull = instance_rHull.component(transBlockName)
        transBlock_rHull.xstar = Param(range(len(transBlock.all_vars)), mutable=True, default=0, within=Reals)
        extendedSpaceCuts = transBlock_rHull.extendedSpaceCuts = Block()
        extendedSpaceCuts.deactivate()
        extendedSpaceCuts.cuts = Constraint(Any)
        var_info = [(v, transBlock_rHull.all_vars[i], transBlock_rHull.xstar[i]) for i, v in enumerate(transBlock.all_vars)]
        return (instance, cuts_obj, instance_rHull, var_info, transBlockName)

    def _create_hull_to_bigm_substitution_map(self, var_info):
        return dict(((id(var_info[i][1]), var_info[i][0]) for i in range(len(var_info))))

    def _create_bigm_to_hull_substitution_map(self, var_info):
        return dict(((id(var_info[i][0]), var_info[i][1]) for i in range(len(var_info))))

    def _get_disaggregated_vars(self, hull):
        disaggregatedVars = ComponentSet()
        for disjunction in hull.component_data_objects(Disjunction, descend_into=(Disjunct, Block)):
            for disjunct in disjunction.disjuncts:
                transBlock = disjunct.transformation_block
                if transBlock is not None:
                    for v in transBlock.disaggregatedVars.component_data_objects(Var):
                        disaggregatedVars.add(v)
        return disaggregatedVars

    def _get_rBigM_obj_and_constraints(self, instance_rBigM):
        rBigM_obj = next(instance_rBigM.component_data_objects(Objective, active=True), None)
        if rBigM_obj is None:
            raise GDP_Error('Cannot apply cutting planes transformation without an active objective in the model!')
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        rBigM_linear_constraints = []
        for cons in instance_rBigM.component_data_objects(Constraint, descend_into=Block, sort=SortComponents.deterministic, active=True):
            body = cons.body
            if body.polynomial_degree() != 1:
                continue
            rBigM_linear_constraints.extend(fme._process_constraint(cons))
        return (rBigM_obj, rBigM_linear_constraints)

    def _generate_cuttingplanes(self, instance_rBigM, cuts_obj, instance_rHull, var_info, transBlockName):
        opt = SolverFactory(self._config.solver)
        stream_solver = self._config.stream_solver
        opt.options = dict(self._config.solver_options)
        improving = True
        prev_obj = None
        epsilon = self._config.minimum_improvement_threshold
        cuts = None
        transBlock_rHull = instance_rHull.component(transBlockName)
        rBigM_obj, rBigM_linear_constraints = self._get_rBigM_obj_and_constraints(instance_rBigM)
        rHull_vars = [i for i in instance_rHull.component_data_objects(Var, descend_into=Block, sort=SortComponents.deterministic)]
        disaggregated_vars = self._get_disaggregated_vars(instance_rHull)
        hull_to_bigm_map = self._create_hull_to_bigm_substitution_map(var_info)
        bigm_to_hull_map = self._create_bigm_to_hull_substitution_map(var_info)
        xhat = ComponentMap()
        while improving:
            results = opt.solve(instance_rBigM, tee=stream_solver, load_solutions=False)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning('Relaxed BigM subproblem did not solve normally. Stopping cutting plane generation.\n\n%s' % (results,))
                return
            instance_rBigM.solutions.load_from(results)
            rBigM_objVal = value(rBigM_obj)
            logger.warning('rBigM objective = %s' % (rBigM_objVal,))
            if transBlock_rHull.component('separation_objective') is None:
                self._add_separation_objective(var_info, transBlock_rHull)
            logger.info('x* is:')
            for x_rbigm, x_hull, x_star in var_info:
                if not x_rbigm.stale:
                    x_star.set_value(x_rbigm.value)
                    x_hull.set_value(x_rbigm.value, skip_validation=True)
                if self.verbose:
                    logger.info('\t%s = %s' % (x_rbigm.getname(fully_qualified=True), x_rbigm.value))
            if prev_obj is None:
                improving = True
            else:
                obj_diff = prev_obj - rBigM_objVal
                improving = abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else abs(obj_diff / prev_obj) > epsilon
            results = opt.solve(instance_rHull, tee=stream_solver, load_solutions=False)
            if verify_successful_solve(results) is not NORMAL:
                logger.warning('Hull separation subproblem did not solve normally. Stopping cutting plane generation.\n\n%s' % (results,))
                return
            instance_rHull.solutions.load_from(results)
            logger.warning('separation problem objective value: %s' % value(transBlock_rHull.separation_objective))
            if self.verbose:
                logger.info('xhat is: ')
            for x_rbigm, x_hull, x_star in var_info:
                xhat[x_rbigm] = value(x_hull)
                if self.verbose:
                    logger.info('\t%s = %s' % (x_hull.getname(fully_qualified=True), x_hull.value))
            if value(transBlock_rHull.separation_objective) < self._config.separation_objective_threshold:
                logger.warning('Separation problem objective below threshold of %s: Stopping cut generation.' % self._config.separation_objective_threshold)
                break
            cuts = self._config.create_cuts(transBlock_rHull, var_info, hull_to_bigm_map, rBigM_linear_constraints, rHull_vars, disaggregated_vars, self._config.norm, self._config.cut_filtering_threshold, self._config.zero_tolerance, self._config.do_integer_arithmetic, self._config.tight_constraint_tolerance)
            if cuts is None:
                logger.warning('Did not generate a valid cut, stopping cut generation.')
                break
            if not improving:
                logger.warning('Difference in relaxed BigM problem objective values from past two iterations is below threshold of %s: Stopping cut generation.' % epsilon)
                break
            for cut in cuts:
                cut_number = len(cuts_obj)
                logger.warning('Adding cut %s to BigM model.' % (cut_number,))
                cuts_obj.add(cut_number, cut)
                if self._config.post_process_cut is not None:
                    self._config.post_process_cut(cuts_obj[cut_number], transBlock_rHull, bigm_to_hull_map, opt, stream_solver, self._config.back_off_problem_tolerance)
            if cut_number + 1 == self._config.max_number_of_cuts:
                logger.warning('Reached maximum number of cuts.')
                break
            prev_obj = rBigM_objVal
            for x_rbigm, x_hull, x_star in var_info:
                x_rbigm.set_value(xhat[x_rbigm], skip_validation=True)

    def _add_transformation_block(self, instance):
        transBlockName = unique_component_name(instance, '_pyomo_gdp_cuttingplane_transformation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        return (transBlockName, transBlock)

    def _add_separation_objective(self, var_info, transBlock_rHull):
        for o in transBlock_rHull.model().component_data_objects(Objective):
            o.deactivate()
        norm = self._config.norm
        to_delete = []
        if norm == 2:
            obj_expr = 0
            for i, (x_rbigm, x_hull, x_star) in enumerate(var_info):
                if not x_rbigm.stale:
                    obj_expr += (x_hull - x_star) ** 2
                else:
                    if self.verbose:
                        logger.info('The variable %s will not be included in the separation problem: It was stale in the rBigM solve.' % x_rbigm.getname(fully_qualified=True))
                    to_delete.append(i)
        elif norm == float('inf'):
            u = transBlock_rHull.u = Var(domain=NonNegativeReals)
            inf_cons = transBlock_rHull.inf_norm_linearization = Constraint(NonNegativeIntegers)
            i = 0
            for j, (x_rbigm, x_hull, x_star) in enumerate(var_info):
                if not x_rbigm.stale:
                    inf_cons[i] = u - x_hull >= -x_star
                    inf_cons[i + 1] = u + x_hull >= x_star
                    i += 2
                else:
                    if self.verbose:
                        logger.info('The variable %s will not be included in the separation problem: It was stale in the rBigM solve.' % x_rbigm.getname(fully_qualified=True))
                    to_delete.append(j)
            self._add_dual_suffix(transBlock_rHull.model())
            obj_expr = u
        for i in sorted(to_delete, reverse=True):
            del var_info[i]
        transBlock_rHull.separation_objective = Objective(expr=obj_expr)

    def _add_dual_suffix(self, rHull):
        dual = rHull.component('dual')
        if dual is None:
            rHull.dual = Suffix(direction=Suffix.IMPORT)
        else:
            if dual.ctype is Suffix:
                return
            rHull.del_component(dual)
            rHull.dual = Suffix(direction=Suffix.IMPORT)
            rHull.add_component(unique_component_name(rHull, 'dual'), dual)