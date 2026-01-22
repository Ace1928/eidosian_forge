import logging
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct, GDP_Error, Disjunction
from pyomo.core import TraversalStrategy, TransformationFactory
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.common.deprecation import deprecated
@TransformationFactory.register('gdp.reclassify', doc='Reclassify Disjuncts to Blocks.')
class HACK_GDP_Disjunct_Reclassifier(Transformation):
    """Reclassify Disjuncts to Blocks.

    HACK: this will reclassify all Disjuncts to Blocks so the current writers
    can find the variables

    """

    @deprecated(msg='The gdp.reclasify transformation has been deprecated in favor of the gdp transformations creating References to variables local to each Disjunct during the transformation. Validation that the model has been completely transformed to an algebraic model has been moved to the check_model_algebraic function in gdp.util.', version='5.7')
    def _apply_to(self, instance, **kwds):
        assert not kwds
        disjunct_generator = instance.component_objects(Disjunct, descend_into=(Block, Disjunct), descent_order=TraversalStrategy.PostfixDFS)
        for disjunct_component in disjunct_generator:
            for disjunct in disjunct_component.values():
                if disjunct.active and self._disjunct_not_relaxed(disjunct) and self._disjunct_on_active_block(disjunct) and self._disjunct_not_fixed_true(disjunct):
                    disjunction_set = {i for i in instance.component_data_objects(Disjunction, descend_into=True, active=None)}
                    active_disjunction_set = {i for i in instance.component_data_objects(Disjunction, descend_into=True, active=True)}
                    disjuncts_in_disjunctions = set()
                    for i in disjunction_set:
                        disjuncts_in_disjunctions.update(i.disjuncts)
                    disjuncts_in_active_disjunctions = set()
                    for i in active_disjunction_set:
                        disjuncts_in_active_disjunctions.update(i.disjuncts)
                    if disjunct not in disjuncts_in_disjunctions:
                        raise GDP_Error('Disjunct "%s" is currently active, but was not found in any Disjunctions. This is generally an error as the model has not been fully relaxed to a pure algebraic form.' % (disjunct.name,))
                    elif disjunct not in disjuncts_in_active_disjunctions:
                        raise GDP_Error('Disjunct "%s" is currently active. While it participates in a Disjunction, that Disjunction is currently deactivated. This is generally an error as the model has not been fully relaxed to a pure algebraic form. Did you deactivate the Disjunction without addressing the individual Disjuncts?' % (disjunct.name,))
                    else:
                        raise GDP_Error('\n                        Reclassifying active Disjunct "%s" as a Block.  This\n                        is generally an error as it indicates that the model\n                        was not completely relaxed before applying the\n                        gdp.reclassify transformation' % (disjunct.name,))
            disjunct_component.parent_block().reclassify_component_type(disjunct_component, Block)
            super(ActiveIndexedComponent, disjunct_component).activate()
            for disjunct in disjunct_component._data.values():
                if self._disjunct_not_relaxed(disjunct):
                    disjunct._deactivate_without_fixing_indicator()
                else:
                    disjunct._activate_without_unfixing_indicator()
                cons_in_disjunct = disjunct.component_objects(Constraint, descend_into=Block, active=True)
                for con in cons_in_disjunct:
                    con.deactivate()

    def _disjunct_not_fixed_true(self, disjunct):
        return not (disjunct.indicator_var.fixed and disjunct.indicator_var.value)

    def _disjunct_not_relaxed(self, disjunct):
        return disjunct.transformation_block is None

    def _disjunct_on_active_block(self, disjunct):
        parent_block = disjunct.parent_block()
        while parent_block is not None:
            if parent_block.ctype is Block and (not parent_block.active):
                return False
            elif parent_block.ctype is Disjunct and (not parent_block.active) and (parent_block.indicator_var.value == False) and parent_block.indicator_var.fixed:
                return False
            else:
                parent_block = parent_block.parent_block()
                continue
        return True