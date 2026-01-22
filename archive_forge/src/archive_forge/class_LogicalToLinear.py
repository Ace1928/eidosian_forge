from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base.boolean_var import _DeprecatedImplicitAssociatedBinaryVariable
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import native_logical_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.util import target_list
@TransformationFactory.register('core.logical_to_linear', doc='Convert logic to linear constraints')
class LogicalToLinear(IsomorphicTransformation):
    """
    Re-encode logical constraints as linear constraints,
    converting Boolean variables to binary.
    """
    CONFIG = ConfigBlock('core.logical_to_linear')
    CONFIG.declare('targets', ConfigValue(default=None, domain=target_list, description='target or list of targets that will be relaxed', doc='\n            This specifies the list of LogicalConstraints to transform, or the\n            list of Blocks or Disjuncts on which to transform all of the\n            LogicalConstraints. Note that if the transformation is done out\n            of place, the list of targets should be attached to the model before it\n            is cloned, and the list will specify the targets on the cloned\n            instance.\n            '))

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        targets = config.targets
        if targets is None:
            targets = (model,)
        new_var_lists = ComponentMap()
        transBlocks = {}
        for t in targets:
            if t.ctype is Block or isinstance(t, _BlockData):
                self._transform_block(t, model, new_var_lists, transBlocks)
            elif t.ctype is LogicalConstraint:
                if t.is_indexed():
                    self._transform_constraint(t, new_var_lists, transBlocks)
                else:
                    self._transform_constraintData(t, new_var_lists, transBlocks)
            else:
                raise RuntimeError("Target '%s' was not a Block, Disjunct, or LogicalConstraint. It was of type %s and can't be transformed." % (t.name, type(t)))

    def _transform_boolean_varData(self, bool_vardata, new_varlists):
        parent_component = bool_vardata.parent_component()
        new_varlist = new_varlists.get(parent_component)
        if new_varlist is None and bool_vardata.get_associated_binary() is None:
            parent_block = bool_vardata.parent_block()
            new_var_list_name = unique_component_name(parent_block, parent_component.local_name + '_asbinary')
            new_varlist = VarList(domain=Binary)
            setattr(parent_block, new_var_list_name, new_varlist)
            new_varlists[parent_component] = new_varlist
        if bool_vardata.get_associated_binary() is None:
            new_binary_vardata = new_varlist.add()
            bool_vardata.associate_binary_var(new_binary_vardata)
            if bool_vardata.value is not None:
                new_binary_vardata.value = int(bool_vardata.value)
            if bool_vardata.fixed:
                new_binary_vardata.fix()

    def _transform_constraint(self, constraint, new_varlists, transBlocks):
        for i in constraint.keys(sort=SortComponents.ORDERED_INDICES):
            self._transform_constraintData(constraint[i], new_varlists, transBlocks)
        constraint.deactivate()

    def _transform_block(self, target_block, model, new_varlists, transBlocks):
        _blocks = target_block.values() if target_block.is_indexed() else (target_block,)
        for block in _blocks:
            for logical_constraint in block.component_objects(ctype=LogicalConstraint, active=True, descend_into=Block):
                self._transform_constraint(logical_constraint, new_varlists, transBlocks)
            for bool_vardata in block.component_data_objects(BooleanVar, descend_into=Block):
                if bool_vardata._associated_binary is None:
                    bool_vardata._associated_binary = _DeprecatedImplicitAssociatedBinaryVariable(bool_vardata)

    def _transform_constraintData(self, logical_constraint, new_varlists, transBlocks):
        for bool_vardata in identify_variables(logical_constraint.expr):
            if bool_vardata.ctype is BooleanVar:
                self._transform_boolean_varData(bool_vardata, new_varlists)
        parent_block = logical_constraint.parent_block()
        xfrm_block = transBlocks.get(parent_block)
        if xfrm_block is None:
            xfrm_block = self._create_transformation_block(parent_block)
            transBlocks[parent_block] = xfrm_block
        new_constrlist = xfrm_block.transformed_constraints
        new_boolvarlist = xfrm_block.augmented_vars
        new_varlist = xfrm_block.augmented_vars_asbinary
        old_boolvarlist_length = len(new_boolvarlist)
        indicator_map = ComponentMap()
        cnf_statements = to_cnf(logical_constraint.body, new_boolvarlist, indicator_map)
        logical_constraint.deactivate()
        num_new = len(new_boolvarlist) - old_boolvarlist_length
        list_o_vars = list(new_boolvarlist.values())
        if num_new:
            for bool_vardata in list_o_vars[-num_new:]:
                new_binary_vardata = new_varlist.add()
                bool_vardata.associate_binary_var(new_binary_vardata)
        for cnf_statement in cnf_statements:
            for linear_constraint in _cnf_to_linear_constraint_list(cnf_statement):
                new_constrlist.add(expr=linear_constraint)
        old_varlist_length = len(new_varlist)
        for indicator_var, special_atom in indicator_map.items():
            for linear_constraint in _cnf_to_linear_constraint_list(special_atom, indicator_var, new_varlist):
                new_constrlist.add(expr=linear_constraint)
        num_new = len(new_varlist) - old_varlist_length
        list_o_vars = list(new_varlist.values())
        if num_new:
            for binary_vardata in list_o_vars[-num_new:]:
                new_bool_vardata = new_boolvarlist.add()
                new_bool_vardata.associate_binary_var(binary_vardata)

    def _create_transformation_block(self, context):
        new_xfrm_block_name = unique_component_name(context, 'logic_to_linear')
        new_xfrm_block = Block(doc='Transformation objects for logic_to_linear')
        setattr(context, new_xfrm_block_name, new_xfrm_block)
        new_xfrm_block.transformed_constraints = ConstraintList()
        new_xfrm_block.augmented_vars = BooleanVarList()
        new_xfrm_block.augmented_vars_asbinary = VarList(domain=Binary)
        return new_xfrm_block