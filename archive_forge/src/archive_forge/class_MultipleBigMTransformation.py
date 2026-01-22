import itertools
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref
@TransformationFactory.register('gdp.mbigm', doc='Relax disjunctive model using big-M terms specific to each disjunct')
class MultipleBigMTransformation(GDP_to_MIP_Transformation, _BigM_MixIn):
    """
    Implements the multiple big-M transformation from [1]. Note that this
    transformation is no different than the big-M transformation for two-
    term disjunctions, but that it may provide a tighter relaxation for
    models containing some disjunctions with three or more terms.

    [1] Francisco Trespalaios and Ignacio E. Grossmann, "Improved Big-M
        reformulation for generalized disjunctive programs," Computers and
        Chemical Engineering, vol. 76, 2015, pp. 98-103.
    """
    CONFIG = ConfigDict('gdp.mbigm')
    CONFIG.declare('targets', ConfigValue(default=None, domain=target_list, description='target or list of targets that will be relaxed', doc='\n        This specifies the list of components to relax. If None (default), the\n        entire model is transformed. Note that if the transformation is done out\n        of place, the list of targets should be attached to the model before it\n        is cloned, and the list will specify the targets on the cloned\n        instance.'))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(default=False, domain=bool, description='Boolean indicating whether or not to transform so that the transformed model will still be valid when fixed Vars are unfixed.', doc='\n        This is only relevant when the transformation will be calculating M\n        values. If True, the transformation will calculate M values assuming\n        that fixed variables will always be fixed to their current values. This\n        means that if a fixed variable is unfixed after transformation, the\n        transformed model is potentially no longer valid. By default, the\n        transformation will assume fixed variables could be unfixed in the\n        future and will use their bounds to calculate the M value rather than\n        their value. Note that this could make for a weaker LP relaxation\n        while the variables remain fixed.\n        '))
    CONFIG.declare('solver', ConfigValue(default=SolverFactory('gurobi'), description='A solver to use to solve the continuous subproblems for calculating the M values'))
    CONFIG.declare('bigM', ConfigValue(default=None, domain=_to_dict, description='Big-M values to use while relaxing constraints', doc="\n        A user-specified dict or ComponentMap mapping tuples of Constraints\n        and Disjuncts to Big-M values valid for relaxing the constraint if\n        the Disjunct is chosen.\n\n        Note: Unlike in the bigm transformation, we require the keys in this\n        mapping specify the components the M value applies to exactly in order\n        to avoid ambiguity. However, if the 'only_mbigm_bound_constraints'\n        option is True, this argument can be used as it would be in the\n        traditional bigm transformation for the non-bound constraints.\n        "))
    CONFIG.declare('reduce_bound_constraints', ConfigValue(default=True, domain=bool, description='Flag indicating whether or not to handle disjunctive constraints that bound a single variable in a single (tighter) constraint, rather than one per Disjunct.', doc='\n        Given the not-uncommon special structure:\n\n        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],\n\n        instead of applying the rote transformation that would create 2*K\n        different constraints in the relaxation, we can write two constraints:\n\n        x >= l_1*y_1 + l_2*y_2 + ... + l_K*y_k\n        x <= u_1*y_1 + u_2*y_2 + ... + u_K*y_K.\n\n        This relaxation is as tight and has fewer constraints. This option is\n        a flag to tell the mbigm transformation to detect this structure and\n        handle it specially. Note that this is a special case of the \'Hybrid\n        Big-M Formulation\' from [2] that takes advantage of the common left-\n        hand side matrix for disjunctive constraints that bound a single\n        variable.\n\n        Note that we do not use user-specified M values for these constraints\n        when this flag is set to True: If tighter bounds exist then they\n        they should be put in the constraints.\n\n        [2] Juan Pablo Vielma, "Mixed Integer Linear Programming Formluation\n            Techniques," SIAM Review, vol. 57, no. 1, 2015, pp. 3-57.\n        '))
    CONFIG.declare('only_mbigm_bound_constraints', ConfigValue(default=False, domain=bool, description='Flag indicating if only bound constraints should be transformed with multiple-bigm, or if all the disjunctive constraints should.', doc='\n        Sometimes it is only computationally advantageous to apply multiple-\n        bigm to disjunctive constraints with the special structure:\n\n        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],\n\n        and transform other disjunctive constraints with the traditional\n        big-M transformation. This flag is used to set the above behavior.\n\n        Note that the reduce_bound_constraints flag must also be True when\n        this flag is set to True.\n        '))
    transformation_name = 'mbigm'

    def __init__(self):
        super().__init__(logger)
        self._arg_list = {}
        self._set_up_expr_bound_visitor()
        self.handlers[Suffix] = self._warn_for_active_suffix

    def _apply_to(self, instance, **kwds):
        self.used_args = ComponentMap()
        with PauseGC():
            try:
                self._apply_to_impl(instance, **kwds)
            finally:
                self._restore_state()
                self.used_args.clear()
                self._arg_list.clear()
                self._expr_bound_visitor.leaf_bounds.clear()
                self._expr_bound_visitor.use_fixed_var_values_as_bounds = False

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        if self._config.assume_fixed_vars_permanent:
            self._bound_visitor.use_fixed_var_values_as_bounds = True
        if self._config.only_mbigm_bound_constraints and (not self._config.reduce_bound_constraints):
            raise GDP_Error("The 'only_mbigm_bound_constraints' option is set to True, but the 'reduce_bound_constraints' option is not. This is not supported--please also set 'reduce_bound_constraints' to True if you only wish to transform the bound constraints with multiple bigm.")
        targets = self._filter_targets(instance)
        self._transform_logical_constraints(instance, targets)
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index(), parent_disjunct=gdp_tree.parent(t), root_disjunct=gdp_tree.root_disjunct(t))
        _warn_for_unused_bigM_args(self._config.bigM, self.used_args, logger)

    def _transform_disjunctionData(self, obj, index, parent_disjunct, root_disjunct):
        if root_disjunct is not None:
            raise GDP_Error("Found nested Disjunction '%s'. The multiple bigm transformation does not support nested GDPs. Please flatten the model before calling the transformation" % obj.name)
        if not obj.xor:
            raise GDP_Error("Cannot do multiple big-M reformulation for Disjunction '%s' with OR constraint.  Must be an XOR!" % obj.name)
        transBlock, algebraic_constraint = self._setup_transform_disjunctionData(obj, root_disjunct)
        arg_Ms = self._config.bigM if self._config.bigM is not None else {}
        active_disjuncts = [disj for disj in obj.disjuncts if disj.active]
        transformed_constraints = set()
        if self._config.reduce_bound_constraints:
            transformed_constraints = self._transform_bound_constraints(active_disjuncts, transBlock, arg_Ms)
        Ms = arg_Ms
        if not self._config.only_mbigm_bound_constraints:
            Ms = transBlock.calculated_missing_m_values = self._calculate_missing_M_values(active_disjuncts, arg_Ms, transBlock, transformed_constraints)
        for cons in transformed_constraints:
            cons.deactivate()
        or_expr = 0
        for disjunct in active_disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            self._transform_disjunct(disjunct, transBlock, active_disjuncts, Ms)
        algebraic_constraint.add(index, or_expr == 1)
        obj._algebraic_constraint = weakref_ref(algebraic_constraint[index])
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, active_disjuncts, Ms):
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)
        self._transform_block_components(obj, obj, active_disjuncts, Ms)
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(self, obj, disjunct, active_disjuncts, Ms):
        relaxationBlock = disjunct._transformation_block()
        constraintMap = relaxationBlock._constraintMap
        transBlock = relaxationBlock.parent_block()
        name = unique_component_name(relaxationBlock, obj.getname(fully_qualified=False))
        newConstraint = Constraint(Any)
        relaxationBlock.add_component(name, newConstraint)
        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue
            if not self._config.only_mbigm_bound_constraints:
                transformed = []
                if c.lower is not None:
                    rhs = sum((Ms[c, disj][0] * disj.indicator_var.get_associated_binary() for disj in active_disjuncts if disj is not disjunct))
                    newConstraint.add((i, 'lb'), c.body - c.lower >= rhs)
                    transformed.append(newConstraint[i, 'lb'])
                if c.upper is not None:
                    rhs = sum((Ms[c, disj][1] * disj.indicator_var.get_associated_binary() for disj in active_disjuncts if disj is not disjunct))
                    newConstraint.add((i, 'ub'), c.body - c.upper <= rhs)
                    transformed.append(newConstraint[i, 'ub'])
                for c_new in transformed:
                    constraintMap['srcConstraints'][c_new] = [c]
                constraintMap['transformedConstraints'][c] = transformed
            else:
                lower = (None, None, None)
                upper = (None, None, None)
                if disjunct not in self._arg_list:
                    self._arg_list[disjunct] = self._get_bigM_arg_list(self._config.bigM, disjunct)
                arg_list = self._arg_list[disjunct]
                lower, upper = self._get_M_from_args(c, Ms, arg_list, lower, upper)
                M = (lower[0], upper[0])
                if c.lower is not None and M[0] is None:
                    M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                    lower = (M[0], None, None)
                if c.upper is not None and M[1] is None:
                    M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                    upper = (M[1], None, None)
                self._add_constraint_expressions(c, i, M, disjunct.indicator_var.get_associated_binary(), newConstraint, constraintMap)
            c.deactivate()

    def _transform_bound_constraints(self, active_disjuncts, transBlock, Ms):
        bounds_cons = ComponentMap()
        lower_bound_constraints_by_var = ComponentMap()
        upper_bound_constraints_by_var = ComponentMap()
        transformed_constraints = set()
        for disj in active_disjuncts:
            for c in disj.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
                repn = generate_standard_repn(c.body)
                if repn.is_linear() and len(repn.linear_vars) == 1:
                    v = repn.linear_vars[0]
                    if v not in bounds_cons:
                        bounds_cons[v] = [{}, {}]
                    M = [None, None]
                    if c.lower is not None:
                        M[0] = (c.lower - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][0]:
                            M[0] = max(M[0], bounds_cons[v][0][disj])
                        bounds_cons[v][0][disj] = M[0]
                        if v in lower_bound_constraints_by_var:
                            lower_bound_constraints_by_var[v].add((c, disj))
                        else:
                            lower_bound_constraints_by_var[v] = {(c, disj)}
                    if c.upper is not None:
                        M[1] = (c.upper - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][1]:
                            M[1] = min(M[1], bounds_cons[v][1][disj])
                        bounds_cons[v][1][disj] = M[1]
                        if v in upper_bound_constraints_by_var:
                            upper_bound_constraints_by_var[v].add((c, disj))
                        else:
                            upper_bound_constraints_by_var[v] = {(c, disj)}
                    transBlock._mbm_values[c, disj] = M
                    transformed_constraints.add(c)
        transformed = transBlock.transformed_bound_constraints
        offset = len(transformed)
        for i, (v, (lower_dict, upper_dict)) in enumerate(bounds_cons.items()):
            lower_rhs = 0
            upper_rhs = 0
            for disj in active_disjuncts:
                relaxationBlock = self._get_disjunct_transformation_block(disj, transBlock)
                if len(lower_dict) > 0:
                    M = lower_dict.get(disj, None)
                    if M is None:
                        M = v.lb
                    if M is None:
                        raise GDP_Error("There is no lower bound for variable '%s', and Disjunct '%s' does not specify one in its constraints. The transformation cannot construct the special bound constraint relaxation without one of these." % (v.name, disj.name))
                    lower_rhs += M * disj.indicator_var.get_associated_binary()
                if len(upper_dict) > 0:
                    M = upper_dict.get(disj, None)
                    if M is None:
                        M = v.ub
                    if M is None:
                        raise GDP_Error("There is no upper bound for variable '%s', and Disjunct '%s' does not specify one in its constraints. The transformation cannot construct the special bound constraint relaxation without one of these." % (v.name, disj.name))
                    upper_rhs += M * disj.indicator_var.get_associated_binary()
            idx = i + offset
            if len(lower_dict) > 0:
                transformed.add((idx, 'lb'), v >= lower_rhs)
                relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'lb']] = []
                for c, disj in lower_bound_constraints_by_var[v]:
                    relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'lb']].append(c)
                    disj.transformation_block._constraintMap['transformedConstraints'][c] = [transformed[idx, 'lb']]
            if len(upper_dict) > 0:
                transformed.add((idx, 'ub'), v <= upper_rhs)
                relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'ub']] = []
                for c, disj in upper_bound_constraints_by_var[v]:
                    relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'ub']].append(c)
                    if c in disj.transformation_block._constraintMap['transformedConstraints']:
                        disj.transformation_block._constraintMap['transformedConstraints'][c].append(transformed[idx, 'ub'])
                    else:
                        disj.transformation_block._constraintMap['transformedConstraints'][c] = [transformed[idx, 'ub']]
        return transformed_constraints

    def _add_transformation_block(self, to_block):
        transBlock, new_block = super()._add_transformation_block(to_block)
        if new_block:
            transBlock._mbm_values = {}
            transBlock.transformed_bound_constraints = Constraint(NonNegativeIntegers, ['lb', 'ub'])
        return (transBlock, new_block)

    def _get_all_var_objects(self, active_disjuncts):
        seen = set()
        for disj in active_disjuncts:
            for constraint in disj.component_data_objects(Constraint, active=True, sort=SortComponents.deterministic, descend_into=Block):
                for var in EXPR.identify_variables(constraint.expr, include_fixed=True):
                    if id(var) not in seen:
                        seen.add(id(var))
                        yield var

    def _calculate_missing_M_values(self, active_disjuncts, arg_Ms, transBlock, transformed_constraints):
        scratch_blocks = {}
        all_vars = list(self._get_all_var_objects(active_disjuncts))
        for disjunct, other_disjunct in itertools.product(active_disjuncts, active_disjuncts):
            if disjunct is other_disjunct:
                continue
            if id(other_disjunct) in scratch_blocks:
                scratch = scratch_blocks[id(other_disjunct)]
            else:
                scratch = scratch_blocks[id(other_disjunct)] = Block()
                other_disjunct.add_component(unique_component_name(other_disjunct, 'scratch'), scratch)
                scratch.obj = Objective(expr=0)
                for v in all_vars:
                    ref = Reference(v)
                    scratch.add_component(unique_component_name(scratch, v.name), ref)
            for constraint in disjunct.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
                if constraint in transformed_constraints:
                    continue
                if (constraint, other_disjunct) in arg_Ms:
                    lower_M, upper_M = _convert_M_to_tuple(arg_Ms[constraint, other_disjunct], constraint, other_disjunct)
                    self.used_args[constraint, other_disjunct] = (lower_M, upper_M)
                else:
                    lower_M, upper_M = (None, None)
                unsuccessful_solve_msg = "Unsuccessful solve to calculate M value to relax constraint '%s' on Disjunct '%s' when Disjunct '%s' is selected." % (constraint.name, disjunct.name, other_disjunct.name)
                if constraint.lower is not None and lower_M is None:
                    if lower_M is None:
                        scratch.obj.expr = constraint.body - constraint.lower
                        scratch.obj.sense = minimize
                        lower_M = self._solve_disjunct_for_M(other_disjunct, scratch, unsuccessful_solve_msg)
                if constraint.upper is not None and upper_M is None:
                    if upper_M is None:
                        scratch.obj.expr = constraint.body - constraint.upper
                        scratch.obj.sense = maximize
                        upper_M = self._solve_disjunct_for_M(other_disjunct, scratch, unsuccessful_solve_msg)
                arg_Ms[constraint, other_disjunct] = (lower_M, upper_M)
                transBlock._mbm_values[constraint, other_disjunct] = (lower_M, upper_M)
        for blk in scratch_blocks.values():
            blk.parent_block().del_component(blk)
        return arg_Ms

    def _solve_disjunct_for_M(self, other_disjunct, scratch_block, unsuccessful_solve_msg):
        solver = self._config.solver
        results = solver.solve(other_disjunct, load_solutions=False)
        if results.solver.termination_condition is TerminationCondition.infeasible:
            if any((s in solver.name for s in _trusted_solvers)):
                logger.debug("Disjunct '%s' is infeasible, deactivating." % other_disjunct.name)
                other_disjunct.deactivate()
                M = 0
            else:
                raise GDP_Error(unsuccessful_solve_msg)
        elif results.solver.termination_condition is not TerminationCondition.optimal:
            raise GDP_Error(unsuccessful_solve_msg)
        else:
            other_disjunct.solutions.load_from(results)
            M = value(scratch_block.obj.expr)
        return M

    def _warn_for_active_suffix(self, suffix, disjunct, active_disjuncts, Ms):
        if suffix.local_name == 'BigM':
            logger.debug("Found active 'BigM' Suffix on '{0}'. The multiple bigM transformation does not currently support specifying M's with Suffixes and is ignoring this Suffix.".format(disjunct.name))
        elif suffix.local_name == 'LocalVars':
            pass
        else:
            raise GDP_Error("Found active Suffix '{0}' on Disjunct '{1}'. The multiple bigM transformation does not support this Suffix.".format(suffix.name, disjunct.name))

    def get_src_constraints(self, transformedConstraint):
        """Return the original Constraints whose transformed counterpart is
        transformedConstraint

        Parameters
        ----------
        transformedConstraint: Constraint, which must be a component on one of
        the BlockDatas in the relaxedDisjuncts Block of
        a transformation block
        """
        return super().get_src_constraint(transformedConstraint)

    def get_all_M_values(self, model):
        """Returns a dictionary mapping each constraint, disjunct pair (where
        the constraint is on a disjunct and the disjunct is in the same
        disjunction as that disjunct) to a tuple: (lower_M_value,
        upper_M_value), where either can be None if the constraint does not
        have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with multiple-BigM
        """
        all_ms = {}
        for disjunction in model.component_data_objects(Disjunction, active=None, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
            if disjunction.algebraic_constraint is not None:
                transBlock = disjunction.algebraic_constraint.parent_block()
                if hasattr(transBlock, '_mbm_values'):
                    all_ms.update(transBlock._mbm_values)
        return all_ms