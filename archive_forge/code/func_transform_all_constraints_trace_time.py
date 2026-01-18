from torch.fx.experimental.migrate_gradual_types.constraint import Conj, Disj, T, F, BinConstraintT, BVar, is_bool_expr
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, TVar, DVar
from torch.fx.experimental.migrate_gradual_types.constraint import Prod, is_algebraic_expression, is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq, op_neq, op_gt, op_lt
from torch.fx.experimental.migrate_gradual_types.operation import op_leq, op_sub, op_div, op_mul, op_mod
from torch.fx.tensor_type import TensorType, Dyn
def transform_all_constraints_trace_time(tracer_root, graph, node, counter=0):
    """
        Takes a node and a graph and generates two sets of constraints.
        One set constraints the node's constraints and another set
        constraints the negation of the node's constraints
        Args:
            tracer_root: the root for getting the module instances
            graph: the graph so far in the tracing process
            node: node that represents a conditional
            counter: variable tracking

        Returns: Two sets of constraints. One with a conjunction with the
        the conditional constraint and the other with a conjunction with
        its negation.

        """
    dimension_dict = {}
    generator = ConstraintGenerator(tracer_root, graph)
    new_constraints, counter = generator.generate_constraints(counter)
    condition_constraint = new_constraints.conjucts[-1]
    new_constraints.conjucts = new_constraints.conjucts[:-1]
    new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)
    assert isinstance(condition_constraint.lhs, BVar)
    assert is_bool_expr(condition_constraint.rhs)
    condition_constraint_rhs = condition_constraint.rhs
    condition_constraint_rhs, counter = iterate_till_fixed_point(condition_constraint_rhs, counter)
    transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)
    transformed_condition_constraint, counter = transform_to_z3(condition_constraint_rhs, counter, dimension_dict)
    negation_transformed_condition_constraint = z3.Not(transformed_condition_constraint)
    return (z3.And([transformed, transformed_condition_constraint]), z3.And([transformed, negation_transformed_condition_constraint]))