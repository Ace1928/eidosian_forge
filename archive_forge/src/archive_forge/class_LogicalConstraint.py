import inspect
import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.boolean_value import as_boolean, BooleanConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set
@ModelComponentFactory.register('General logical constraints.')
class LogicalConstraint(ActiveIndexedComponent):
    """
    This modeling component defines a logical constraint using a
    rule function.

    Constructor arguments:
        expr
            A Pyomo expression for this logical constraint
        rule
            A function that is used to construct logical constraints
        doc
            A text string describing this component
        name
            A name for this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to
            construct a model instance
        rule
           The rule used to initialize the logical constraint(s)

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index_set
            The set of valid indices
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """
    _ComponentDataClass = _GeneralLogicalConstraintData

    class Infeasible(object):
        pass
    Feasible = ActiveIndexedComponent.Skip
    NoConstraint = ActiveIndexedComponent.Skip
    Violated = Infeasible
    Satisfied = Feasible

    def __new__(cls, *args, **kwds):
        if cls != LogicalConstraint:
            return super(LogicalConstraint, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarLogicalConstraint.__new__(ScalarLogicalConstraint)
        else:
            return IndexedLogicalConstraint.__new__(IndexedLogicalConstraint)

    def __init__(self, *args, **kwargs):
        self.rule = kwargs.pop('rule', None)
        self._init_expr = kwargs.pop('expr', None)
        kwargs.setdefault('ctype', LogicalConstraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    def _setitem_impl(self, index, obj, value):
        if self._check_skip_add(index, value) is None:
            del self[index]
            return None
        else:
            obj.set_value(value)
            return obj

    def _setitem_when_not_present(self, index, value):
        if self._check_skip_add(index, value) is None:
            return None
        else:
            return super(LogicalConstraint, self)._setitem_when_not_present(index=index, value=value)

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical constraint.
        """
        if is_debug_set(logger):
            logger.debug('Constructing logical constraint %s' % self.name)
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()
        _init_expr = self._init_expr
        _init_rule = self.rule
        self._init_expr = None
        if _init_rule is None and _init_expr is None:
            return
        _self_parent = self._parent()
        if not self.is_indexed():
            if _init_rule is None:
                tmp = _init_expr
            else:
                try:
                    tmp = _init_rule(_self_parent)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error('Rule failed when generating expression for logical constraint %s:\n%s: %s' % (self.name, type(err).__name__, err))
                    raise
            self._setitem_when_not_present(None, tmp)
        else:
            if _init_expr is not None:
                raise IndexError("LogicalConstraint '%s': Cannot initialize multiple indices of a logical constraint with a single expression" % (self.name,))
            for ndx in self._index_set:
                try:
                    tmp = apply_indexed_rule(self, _init_rule, _self_parent, ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error('Rule failed when generating expression for logical constraint %s with index %s:\n%s: %s' % (self.name, str(ndx), type(err).__name__, err))
                    raise
                self._setitem_when_not_present(ndx, tmp)
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return ([('Size', len(self)), ('Index', self._index_set if self.is_indexed() else None), ('Active', self.active)], self.items(), ('Body', 'Active'), lambda k, v: [v.body, v.active])

    def display(self, prefix='', ostream=None):
        """
        Print component state information

        This duplicates logic in Component.pprint()
        """
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab = '    '
        ostream.write(prefix + self.local_name + ' : ')
        ostream.write('Size=' + str(len(self)))
        ostream.write('\n')
        tabular_writer(ostream, prefix + tab, ((k, v) for k, v in self._data.items() if v.active), ('Body',), lambda k, v: [v.body()])

    def _check_skip_add(self, index, expr):
        _expr_type = expr.__class__
        if expr is None:
            raise ValueError(_rule_returned_none_error % (_get_indexed_component_data_name(self, index),))
        if expr is True:
            raise ValueError("LogicalConstraint '%s' is always True." % (_get_indexed_component_data_name(self, index),))
        if expr is False:
            raise ValueError("LogicalConstraint '%s' is always False." % (_get_indexed_component_data_name(self, index),))
        if _expr_type is tuple and len(expr) == 1:
            if expr is LogicalConstraint.Skip:
                return None
            if expr is LogicalConstraint.Infeasible:
                raise ValueError("LogicalConstraint '%s' cannot be passed 'Infeasible' as a value." % (_get_indexed_component_data_name(self, index),))
        return expr