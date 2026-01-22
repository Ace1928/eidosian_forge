from functools import wraps
from pyomo.common.collections import ComponentMap
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.network import Port
from weakref import ref as weakref_ref
class GDP_to_MIP_Transformation(Transformation):
    """
    Base class for transformations from GDP to MIP
    """

    def __init__(self, logger):
        """Initialize transformation object."""
        super(GDP_to_MIP_Transformation, self).__init__()
        self.logger = logger
        self.handlers = {Constraint: self._transform_constraint, Var: False, BooleanVar: False, Connector: False, Expression: False, Suffix: False, Param: False, Set: False, SetOf: False, RangeSet: False, Disjunction: False, Disjunct: self._warn_for_active_disjunct, Block: False, ExternalFunction: False, Port: False}
        self._generate_debug_messages = False
        self._transformation_blocks = {}
        self._algebraic_constraints = {}

    def _restore_state(self):
        self._transformation_blocks.clear()
        self._algebraic_constraints.clear()
        if hasattr(self, '_config'):
            del self._config

    def _process_arguments(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance' must be a ConcreteModel, Block, or Disjunct (in the case of nested disjunctions)." % (instance.name, instance.ctype))
        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)
        self._generate_debug_messages = is_debug_set(self.logger)

    def _transform_logical_constraints(self, instance, targets):
        disj_targets = []
        for t in targets:
            disj_datas = t.values() if t.is_indexed() else [t]
            if t.ctype is Disjunct:
                disj_targets.extend(disj_datas)
            if t.ctype is Disjunction:
                disj_targets.extend([d for disjunction in disj_datas for d in disjunction.disjuncts])
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(instance, targets=[blk for blk in targets if blk.ctype is Block] + disj_targets)

    def _filter_targets(self, instance):
        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        def _filter_inactive(targets):
            for t in targets:
                if not t.active:
                    self.logger.warning(f'GDP.{self.transformation_name} transformation passed a deactivated target ({t.name}). Skipping.')
                else:
                    yield t
        return list(_filter_inactive(targets))

    def _get_gdp_tree_from_targets(self, instance, targets):
        knownBlocks = {}
        return get_gdp_tree(targets, instance)

    def _add_transformation_block(self, to_block):
        if to_block in self._transformation_blocks:
            return (self._transformation_blocks[to_block], False)
        transBlockName = unique_component_name(to_block, '_pyomo_gdp_%s_reformulation' % self.transformation_name)
        self._transformation_blocks[to_block] = transBlock = Block()
        to_block.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = _TransformedDisjunct(NonNegativeIntegers)
        return (transBlock, True)

    def _add_xor_constraint(self, disjunction, transBlock):
        if disjunction in self._algebraic_constraints:
            return self._algebraic_constraints[disjunction]
        if disjunction.is_indexed():
            orC = Constraint(Any)
        else:
            orC = Constraint()
        orCname = unique_component_name(transBlock, disjunction.getname(fully_qualified=False) + '_xor')
        transBlock.add_component(orCname, orC)
        self._algebraic_constraints[disjunction] = orC
        return orC

    def _setup_transform_disjunctionData(self, obj, root_disjunct):
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is likely indicative of a modeling error." % obj.name)
        trans_block, new_block = self._add_transformation_block(obj.parent_block())
        algebraic_constraint = self._add_xor_constraint(obj.parent_component(), trans_block)
        if root_disjunct is not None:
            trans_block, new_block = self._add_transformation_block(root_disjunct.parent_block())
        return (trans_block, algebraic_constraint)

    def _get_disjunct_transformation_block(self, disjunct, transBlock):
        if disjunct.transformation_block is not None:
            return disjunct.transformation_block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        relaxationBlock.transformedConstraints = Constraint(Any)
        relaxationBlock.localVarReferences = Block()
        relaxationBlock._constraintMap = {'srcConstraints': ComponentMap(), 'transformedConstraints': ComponentMap()}
        disjunct._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._src_disjunct = weakref_ref(disjunct)
        return relaxationBlock

    def _transform_block_components(self, block, disjunct, *args):
        varRefBlock = disjunct._transformation_block().localVarReferences
        for v in block.component_objects(Var, descend_into=Block, active=None):
            varRefBlock.add_component(unique_component_name(varRefBlock, v.getname(fully_qualified=False)), Reference(v))
        for obj in block.component_objects(active=True, descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error('No %s transformation handler registered for modeling components of type %s. If your disjuncts contain non-GDP Pyomo components that require transformation, please transform them first.' % (self.transformation_name, obj.ctype))
                continue
            handler(obj, disjunct, *args)

    def _transform_constraint(self, obj, disjunct, *args):
        raise NotImplementedError("Class %s failed to implement '_transform_constraint'" % self.__class__)

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct, *args):
        _warn_for_active_disjunct(innerdisjunct, outerdisjunct)

    @wraps(get_src_disjunct)
    def get_src_disjunct(self, transBlock):
        return get_src_disjunct(transBlock)

    @wraps(get_src_disjunction)
    def get_src_disjunction(self, xor_constraint):
        return get_src_disjunction(xor_constraint)

    @wraps(get_src_constraint)
    def get_src_constraint(self, transformedConstraint):
        return get_src_constraint(transformedConstraint)

    @wraps(get_transformed_constraints)
    def get_transformed_constraints(self, srcConstraint):
        return get_transformed_constraints(srcConstraint)