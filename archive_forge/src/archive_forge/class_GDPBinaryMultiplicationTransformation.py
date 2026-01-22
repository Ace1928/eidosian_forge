from .gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core.base import TransformationFactory
from pyomo.core.util import target_list
from pyomo.gdp import Disjunction
from weakref import ref as weakref_ref
import logging
@TransformationFactory.register('gdp.binary_multiplication', doc='Reformulate the GDP as an MINLP by multiplying f(x) <= 0 by y to get f(x) * y <= 0 where y is the binary corresponding to the Boolean indicator var of the Disjunct containing f(x) <= 0.')
class GDPBinaryMultiplicationTransformation(GDP_to_MIP_Transformation):
    CONFIG = ConfigDict('gdp.binary_multiplication')
    CONFIG.declare('targets', ConfigValue(default=None, domain=target_list, description='target or list of targets that will be transformed', doc='\n\n        This specifies the list of components to transform. If None (default), the\n        entire model is transformed. Note that if the transformation is done out\n        of place, the list of targets should be attached to the model before it\n        is cloned, and the list will specify the targets on the cloned\n        instance.'))
    transformation_name = 'binary_multiplication'

    def __init__(self):
        super().__init__(logger)

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        targets = self._filter_targets(instance)
        self._transform_logical_constraints(instance, targets)
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index(), parent_disjunct=gdp_tree.parent(t), root_disjunct=gdp_tree.root_disjunct(t))

    def _transform_disjunctionData(self, obj, index, parent_disjunct=None, root_disjunct=None):
        transBlock, xorConstraint = self._setup_transform_disjunctionData(obj, root_disjunct)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.binary_indicator_var
            self._transform_disjunct(disjunct, transBlock)
        if obj.xor:
            xorConstraint[index] = or_expr == 1
        else:
            xorConstraint[index] = or_expr >= 1
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock):
        if not obj.active:
            return
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)
        self._transform_block_components(obj, obj)
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(self, obj, disjunct):
        transBlock = disjunct._transformation_block()
        constraintMap = transBlock._constraintMap
        disjunctionRelaxationBlock = transBlock.parent_block()
        newConstraint = transBlock.transformedConstraints
        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue
            self._add_constraint_expressions(c, i, disjunct.binary_indicator_var, newConstraint, constraintMap)
            c.deactivate()

    def _add_constraint_expressions(self, c, i, indicator_var, newConstraint, constraintMap):
        unique = len(newConstraint)
        name = c.local_name + '_%s' % unique
        transformed = constraintMap['transformedConstraints'][c] = []
        lb, ub = (c.lower, c.upper)
        if (c.equality or lb is ub) and lb is not None:
            newConstraint.add((name, i, 'eq'), (c.body - lb) * indicator_var == 0)
            transformed.append(newConstraint[name, i, 'eq'])
            constraintMap['srcConstraints'][newConstraint[name, i, 'eq']] = c
        else:
            if lb is not None:
                newConstraint.add((name, i, 'lb'), 0 <= (c.body - lb) * indicator_var)
                transformed.append(newConstraint[name, i, 'lb'])
                constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
            if ub is not None:
                newConstraint.add((name, i, 'ub'), (c.body - ub) * indicator_var <= 0)
                transformed.append(newConstraint[name, i, 'ub'])
                constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c