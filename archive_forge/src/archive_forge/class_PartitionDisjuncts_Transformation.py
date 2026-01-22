from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref
from math import floor
import logging
@TransformationFactory.register('gdp.partition_disjuncts', doc='Reformulates a convex disjunctive model into a new GDP by splitting additively separable constraints on P sets of variables')
@document_kwargs_from_configdict('CONFIG')
class PartitionDisjuncts_Transformation(Transformation):
    """
    Transform disjunctive model to equivalent disjunctive model (with
    potentially tighter hull relaxation) by taking the "P-split" formulation
    from Kronqvist et al. 2021 [1]. In each Disjunct, convex and additively
    separable constraints are split into separate constraints by introducing
    auxiliary variables that upperbound the subexpressions created by the split.
    Increasing the number of partitions can result in tighter hull relaxations,
    but at the cost of larger model sizes.

    The transformation will create a new Block with a unique name beginning
    "_pyomo_gdp_partition_disjuncts_reformulation".
    The Block will have new Disjunct objects, each corresponding to one of the
    Disjuncts being transformed. These will have the transformed constraints on
    them, and be in new Disjunctions, each corresponding to one of the
    originals. In addition, the auxiliary variables and the partitioned
    constraints will be declared on this Block, as well as LogicalConstraints
    linking the original indicator_vars with the ones of the transformed
    Disjuncts. All original GDP components that were transformed will be
    deactivated.

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate
            Relaxations between big-M and Convex Hull Reformulations," 2021.

    """
    CONFIG = ConfigBlock('gdp.partition_disjuncts')
    CONFIG.declare('targets', ConfigValue(default=None, domain=target_list, description='target or list of targets that will be relaxed', doc='\n        Specifies the target or list of targets to relax as either a\n        component or a list of components. \n\n        If None (default), the entire model is transformed. Note that if the \n        transformation is done out of place, the list of targets should be \n        attached to the model before it is cloned, and the list will specify \n        the targets on the cloned instance.\n        '))
    CONFIG.declare('variable_partitions', ConfigValue(default=None, domain=_to_dict, description='Set of sets of variables which define valid partitions\n        (i.e., the constraints are additively separable across these\n        partitions). These can be specified globally (for all active\n        Disjunctions), or by Disjunction.', doc="\n        Specified variable partitions, either globally or per Disjunction.\n\n        Expects either a set of disjoint ComponentSets whose union is all the\n        variables that appear in all Disjunctions or a mapping from each active\n        Disjunction to a set of disjoint ComponentSets whose union is the set\n        of variables that appear in that Disjunction. In either case, if any\n        constraints in the Disjunction are only partially additively separable,\n        these sets must be a valid partition so that these constraints are\n        additively separable with respect to this partition. To specify a\n        default partition for Disjunctions that do not appear as keys in the\n        map, map the partition to 'None.'\n\n        Last, note that in the case of constraints containing partially\n        additively separable functions, it is required that the user specify\n        the variable partition(s).\n        "))
    CONFIG.declare('num_partitions', ConfigValue(default=None, domain=_to_dict, description='Number of partitions of variables, if variable_partitions\n        is not specified. Can be specified separately for specific Disjunctions\n        if desired.', doc='\n        Either a single value so that all Disjunctions will have variables\n        partitioned into P sets, or a map of Disjunctions to a value of P\n        for each active Disjunction. Mapping None to a value of P will specify\n        the default value of P to use if the value for a given Disjunction\n        is not explicitly specified.\n\n        Note that if any constraints contain partially additively separable\n        functions, the partitions for the Disjunctions with these Constraints\n        must be specified in the variable_partitions argument.\n        '))
    CONFIG.declare('variable_partitioning_method', ConfigValue(default=arbitrary_partition, domain=_to_dict, description='Method to partition the variables. By default, the\n        partitioning will be done arbitrarily.', doc="\n        A function which takes a Disjunction object and a number P and return\n        a valid partitioning of the variables that appear in the disjunction\n        into P partitions.\n\n        Note that you must give a value for 'P' if you are using this method\n        to calculate partitions.\n\n        Note that if any constraints contain partially additively separable\n        functions, the partitions for the Disjunctions cannot be calculated\n        automatically. Please specify the partitions for the Disjunctions with\n        these Constraints in the variable_partitions argument.\n        "))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(default=False, domain=bool, description='Boolean indicating whether or not to transform so that\n        the transformed model will still be valid when fixed Vars are\n        unfixed.', doc="\n        If True, the transformation will create a correct model even if fixed\n        variables are later unfixed. That is, bounds will be calculated based\n        on fixed variables' bounds, not their values. However, if fixed\n        variables will never be unfixed, a possibly tighter model will result,\n        and fixed variables need not have bounds.\n\n        Note that this has no effect on fixed BooleanVars, including the\n        indicator variables of Disjuncts. The transformation is always correct\n        whether or not these remain fixed.\n        "))
    CONFIG.declare('compute_bounds_method', ConfigValue(default=compute_fbbt_bounds, description='Function that takes an expression, a Block containing\n        the global constraints of the original problem, and a configured\n        solver, and returns both a lower and upper bound for the expression.', doc='\n        Callback for computing bounds on expressions, in order to bound\n        the auxiliary variables created by the transformation. \n\n        Some pre-implemented options include\n            * compute_fbbt_bounds (the default), and\n            * compute_optimal_bounds\n        or you can write your own callback which accepts an Expression object,\n        a model containing the variables and global constraints of the original\n        instance, and a configured solver and returns a tuple (LB, UB) where\n        either element can be None if no valid bound could be found.\n        '))
    CONFIG.declare('compute_bounds_solver', ConfigValue(default=None, description="Solver object to pass to compute_bounds_method.\n        This is required if you are using 'compute_optimal_bounds'.", doc="\n        Configured solver object for use in the compute_bounds_method.\n\n        In particular, if compute_bounds_method is 'compute_optimal_bounds',\n        this will be used to solve the subproblems, so needs to handle\n        non-convex problems if any Disjunctions contain nonlinear constraints.\n        "))

    def __init__(self):
        super(PartitionDisjuncts_Transformation, self).__init__()
        self.handlers = {Constraint: self._transform_constraint, Var: False, BooleanVar: False, Connector: False, Expression: False, Suffix: False, Param: False, Set: False, SetOf: False, RangeSet: False, Disjunct: self._warn_for_active_disjunct, Block: False, ExternalFunction: False, Port: False}

    def _apply_to(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance' must be a ConcreteModel, Block, or Disjunct (in the case of nested disjunctions)." % (instance.name, instance.ctype))
        try:
            self._config = self.CONFIG(kwds.pop('options', {}))
            self._config.set_value(kwds)
            self._transformation_blocks = {}
            if not self._config.assume_fixed_vars_permanent:
                fixed_vars = ComponentMap()
                for v in get_vars_from_components(instance, Constraint, include_fixed=True, active=True, descend_into=(Block, Disjunct)):
                    if v.fixed:
                        fixed_vars[v] = value(v)
                        v.fixed = False
            self._apply_to_impl(instance)
        finally:
            if not self._config.assume_fixed_vars_permanent:
                for v, val in fixed_vars.items():
                    v.fix(val)
            del self._config
            del self._transformation_blocks

    def _apply_to_impl(self, instance):
        self.variable_partitions = self._config.variable_partitions if self._config.variable_partitions is not None else {}
        self.partitioning_method = self._config.variable_partitioning_method
        global_constraints = ConcreteModel()
        for cons in instance.component_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
            global_constraints.add_component(unique_component_name(global_constraints, cons.getname(fully_qualified=True)), Reference(cons))
        for var in instance.component_objects(Var, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
            global_constraints.add_component(unique_component_name(global_constraints, var.getname(fully_qualified=True)), Reference(var))
        self._global_constraints = global_constraints
        targets = self._config.targets
        knownBlocks = {}
        if targets is None:
            targets = (instance,)
        targets = self._preprocess_targets(targets, instance, knownBlocks)
        for t in targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index())
            else:
                self._transform_blockData(t)

    def _preprocess_targets(self, targets, instance, knownBlocks):
        gdp_tree = get_gdp_tree(targets, instance, knownBlocks)
        preprocessed_targets = []
        for node in gdp_tree.vertices:
            if gdp_tree.in_degree(node) == 0:
                preprocessed_targets.append(node)
        return preprocessed_targets

    def _get_transformation_block(self, block):
        if self._transformation_blocks.get(block) is not None:
            return self._transformation_blocks[block]
        self._transformation_blocks[block] = transformation_block = Block()
        block.add_component(unique_component_name(block, '_pyomo_gdp_partition_disjuncts_reformulation'), transformation_block)
        transformation_block.indicator_var_equalities = LogicalConstraint(NonNegativeIntegers)
        return transformation_block

    def _transform_blockData(self, obj):
        to_transform = []
        for disjunction in obj.component_data_objects(Disjunction, active=True, sort=SortComponents.deterministic, descend_into=Block):
            to_transform.append(disjunction)
        for disjunction in to_transform:
            self._transform_disjunctionData(disjunction, disjunction.index())

    def _transform_disjunctionData(self, obj, idx, transBlock=None, transformed_parent_disjunct=None):
        if not obj.active:
            return
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is likely indicative of a modeling error." % obj.getname(fully_qualified=True))
        if transBlock is None and transformed_parent_disjunct is not None:
            transBlock = self._get_transformation_block(transformed_parent_disjunct)
        if transBlock is None:
            transBlock = self._get_transformation_block(obj.parent_block())
        variable_partitions = self.variable_partitions
        partition_method = self.partitioning_method
        partition = variable_partitions.get(obj)
        if partition is None:
            partition = variable_partitions.get(None)
            if partition is None:
                method = partition_method.get(obj)
                if method is None:
                    method = partition_method.get(None)
                method = method if method is not None else arbitrary_partition
                if self._config.num_partitions is None:
                    P = None
                else:
                    P = self._config.num_partitions.get(obj)
                    if P is None:
                        P = self._config.num_partitions.get(None)
                if P is None:
                    raise GDP_Error('No value for P was given for disjunction %s! Please specify a value of P (number of partitions), if you do not specify the partitions directly.' % obj.name)
                partition = method(obj, P)
        partition = [ComponentSet(var_list) for var_list in partition]
        transformed_disjuncts = []
        for disjunct in obj.disjuncts:
            transformed_disjunct = self._transform_disjunct(disjunct, partition, transBlock)
            if transformed_disjunct is not None:
                transformed_disjuncts.append(transformed_disjunct)
                transBlock.indicator_var_equalities[len(transBlock.indicator_var_equalities)] = disjunct.indicator_var.equivalent_to(transformed_disjunct.indicator_var)
        transformed_disjunction = Disjunction(expr=[disj for disj in transformed_disjuncts])
        transBlock.add_component(unique_component_name(transBlock, obj.getname(fully_qualified=True)), transformed_disjunction)
        obj._algebraic_constraint = weakref_ref(transformed_disjunction)
        obj.deactivate()

    def _get_leq_constraints(self, cons):
        constraints = []
        if cons.lower is not None:
            constraints.append((-cons.body, -cons.lower))
        if cons.upper is not None:
            constraints.append((cons.body, cons.upper))
        return constraints

    def _transform_disjunct(self, disjunct, partition, transBlock):
        if not disjunct.active:
            if disjunct.indicator_var.is_fixed():
                if not value(disjunct.indicator_var):
                    return
                else:
                    raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is fixed to %s. This makes no sense." % (disjunct.name, value(disjunct.indicator_var)))
            if disjunct._transformation_block is None:
                raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is not fixed and the disjunct does not appear to have been relaxed. This makes no sense. (If the intent is to deactivate the disjunct, fix its indicator_var to False.)" % (disjunct.name,))
        if disjunct._transformation_block is not None:
            raise GDP_Error("The disjunct '%s' has been transformed, but a disjunction it appears in has not. Putting the same disjunct in multiple disjunctions is not supported." % disjunct.name)
        transformed_disjunct = Disjunct()
        disjunct._transformation_block = weakref_ref(transformed_disjunct)
        transBlock.add_component(unique_component_name(transBlock, disjunct.getname(fully_qualified=True)), transformed_disjunct)
        if disjunct.indicator_var.fixed:
            transformed_disjunct.indicator_var.fix(value(disjunct.indicator_var))
        for disjunction in disjunct.component_data_objects(Disjunction, active=True, sort=SortComponents.deterministic, descend_into=Block):
            self._transform_disjunctionData(disjunction, disjunction.index(), None, transformed_disjunct)
        for var in disjunct.component_objects(Var, descend_into=Block, active=None):
            transformed_disjunct.add_component(unique_component_name(transformed_disjunct, var.getname(fully_qualified=True)), Reference(var))
        logical_constraints = LogicalConstraintList()
        transformed_disjunct.add_component(unique_component_name(transformed_disjunct, 'logical_constraints'), logical_constraints)
        for cons in disjunct.component_data_objects(LogicalConstraint, descend_into=Block, active=None):
            logical_constraints.add(cons.expr)
            cons.deactivate()
        for obj in disjunct.component_data_objects(active=True, sort=SortComponents.deterministic, descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error('No partition_disjuncts transformation handler registered for modeling components of type %s. If your disjuncts contain non-GDP Pyomo components that require transformation, please transform them first.' % obj.ctype)
                continue
            handler(obj, disjunct, transformed_disjunct, transBlock, partition)
        disjunct._deactivate_without_fixing_indicator()
        return transformed_disjunct

    def _transform_constraint(self, cons, disjunct, transformed_disjunct, transBlock, partition):
        instance = disjunct.model()
        cons_name = cons.getname(fully_qualified=True)
        transformed_constraint = Constraint(NonNegativeIntegers)
        transformed_disjunct.add_component(unique_component_name(transformed_disjunct, cons_name), transformed_constraint)
        aux_vars = Var(NonNegativeIntegers, dense=False)
        transformed_disjunct.add_component(unique_component_name(transformed_disjunct, cons_name + '_aux_vars'), aux_vars)
        split_constraints = Constraint(NonNegativeIntegers)
        transBlock.add_component(unique_component_name(transBlock, cons_name + '_split_constraints'), split_constraints)
        leq_constraints = self._get_leq_constraints(cons)
        for body, rhs in leq_constraints:
            repn = generate_standard_repn(body, compute_values=True)
            nonlinear_repn = None
            if repn.nonlinear_expr is not None:
                nonlinear_repn = _generate_additively_separable_repn(repn.nonlinear_expr)
            split_exprs = []
            split_aux_vars = []
            vars_not_accounted_for = ComponentSet((v for v in EXPR.identify_variables(body, include_fixed=False)))
            vars_accounted_for = ComponentSet()
            for idx, var_list in enumerate(partition):
                split_exprs.append(0)
                expr = split_exprs[-1]
                for i, v in enumerate(repn.linear_vars):
                    if v in var_list:
                        expr += repn.linear_coefs[i] * v
                        vars_accounted_for.add(v)
                for i, (v1, v2) in enumerate(repn.quadratic_vars):
                    if v1 in var_list:
                        if v2 not in var_list:
                            raise GDP_Error("Variables '%s' and '%s' are multiplied in Constraint '%s', but they are in different partitions! Please ensure that all the constraints in the disjunction are additively separable with respect to the specified partition." % (v1.name, v2.name, cons.name))
                        expr += repn.quadratic_coefs[i] * v1 * v2
                        vars_accounted_for.add(v1)
                        vars_accounted_for.add(v2)
                if nonlinear_repn is not None:
                    for i, expr_var_set in enumerate(nonlinear_repn['nonlinear_vars']):
                        if all((v in var_list for v in list(expr_var_set))):
                            expr += nonlinear_repn['nonlinear_exprs'][i]
                            for var in expr_var_set:
                                vars_accounted_for.add(var)
                        elif len(ComponentSet(expr_var_set) & var_list) != 0:
                            raise GDP_Error("Variables which appear in the expression %s are in different partitions, but this expression doesn't appear additively separable. Please expand it if it is additively separable or, more likely, ensure that all the constraints in the disjunction are additively separable with respect to the specified partition. If you did not specify a partition, only a value of P, note that to automatically partition the variables, we assume all the expressions are additively separable." % nonlinear_repn['nonlinear_exprs'][i])
                expr_lb, expr_ub = self._config.compute_bounds_method(expr, self._global_constraints, self._config.compute_bounds_solver)
                if expr_lb is None or expr_ub is None:
                    raise GDP_Error("Expression %s from constraint '%s' is unbounded! Please ensure all variables that appear in the constraint are bounded or specify compute_bounds_method=compute_optimal_bounds if the expression is bounded by the global constraints." % (expr, cons.name))
                if type(expr) is not int or expr != 0:
                    aux_var = aux_vars[len(aux_vars)]
                    aux_var.setlb(expr_lb)
                    aux_var.setub(expr_ub)
                    split_aux_vars.append(aux_var)
                    split_constraints[len(split_constraints)] = expr <= aux_var
            if len(vars_accounted_for) < len(vars_not_accounted_for):
                orphans = vars_not_accounted_for - vars_accounted_for
                orphan_string = ''
                for v in orphans:
                    orphan_string += "'%s', " % v.name
                orphan_string = orphan_string[:-2]
                raise GDP_Error("Partition specified for disjunction containing Disjunct '%s' does not include all the variables that appear in the disjunction. The following variables are not assigned to any part of the partition: %s" % (disjunct.name, orphan_string))
            transformed_constraint[len(transformed_constraint)] = sum((v for v in split_aux_vars)) <= rhs - repn.constant
        cons.deactivate()

    def _warn_for_active_disjunct(self, disjunct, parent_disjunct, transformed_parent_disjunct, transBlock, partition):
        _warn_for_active_disjunct(disjunct, parent_disjunct)