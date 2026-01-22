import logging, sys
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr import identify_variables
from pyomo.core.base.label import alphanum_label_from_name
from pyomo.network.util import create_var, tighten_var_domain
class _PortData(ComponentData):
    """
    This class defines the data for a single Port

    Attributes
    ----------
        vars:`dict`
            A dictionary mapping added names to variables
    """
    __slots__ = ('vars', '_arcs', '_sources', '_dests', '_rules', '_splitfracs')
    __autoslot_mappers__ = {'_arcs': AutoSlots.weakref_sequence_mapper, '_sources': AutoSlots.weakref_sequence_mapper, '_dests': AutoSlots.weakref_sequence_mapper}

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self.vars = {}
        self._arcs = []
        self._sources = []
        self._dests = []
        self._rules = {}
        self._splitfracs = ComponentMap()

    def __getattr__(self, name):
        """Returns `self.vars[name]` if it exists"""
        if name in self.vars:
            return self.vars[name]
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def arcs(self, active=None):
        """A list of Arcs in which this Port is a member"""
        return self._collect_ports(active, self._arcs)

    def sources(self, active=None):
        """A list of Arcs in which this Port is a destination"""
        return self._collect_ports(active, self._sources)

    def dests(self, active=None):
        """A list of Arcs in which this Port is a source"""
        return self._collect_ports(active, self._dests)

    def _collect_ports(self, active, port_list):
        if active is None:
            return [_a() for _a in port_list]
        tmp = []
        for _a in port_list:
            a = _a()
            if a.active == active:
                tmp.append(a)
        return tmp

    def set_value(self, value):
        """Cannot specify the value of a port"""
        raise ValueError("Cannot specify the value of a port: '%s'" % self.name)

    def polynomial_degree(self):
        """Returns the maximum polynomial degree of all port members"""
        ans = 0
        for v in self.iter_vars():
            tmp = v.polynomial_degree()
            if tmp is None:
                return None
            ans = max(ans, tmp)
        return ans

    def is_fixed(self):
        """Return True if all vars/expressions in the Port are fixed"""
        return all((v.is_fixed() for v in self.iter_vars()))

    def is_potentially_variable(self):
        """Return True as ports may (should!) contain variables"""
        return True

    def is_binary(self):
        """Return True if all variables in the Port are binary"""
        return len(self) and all((v.is_binary() for v in self.iter_vars(expr_vars=True)))

    def is_integer(self):
        """Return True if all variables in the Port are integer"""
        return len(self) and all((v.is_integer() for v in self.iter_vars(expr_vars=True)))

    def is_continuous(self):
        """Return True if all variables in the Port are continuous"""
        return len(self) and all((v.is_continuous() for v in self.iter_vars(expr_vars=True)))

    def add(self, var, name=None, rule=None, **kwds):
        """
        Add `var` to this Port, casting it to a Pyomo numeric if necessary

        Arguments
        ---------
            var
                A variable or some `NumericValue` like an expression
            name: `str`
                Name to associate with this member of the Port
            rule: `function`
                Function implementing the desired expansion procedure
                for this member. `Port.Equality` by default, other
                options include `Port.Extensive`. Customs are allowed.
            kwds
                Keyword arguments that will be passed to rule
        """
        if var is not None:
            try:
                var.is_indexed()
            except AttributeError:
                var = as_numeric(var)
        if name is None:
            name = var.local_name
        if name in self.vars and self.vars[name] is not None:
            logger.warning("Implicitly replacing variable '%s' in Port '%s'.\nTo avoid this warning, use Port.remove() first." % (name, self.name))
        self.vars[name] = var
        if rule is None:
            rule = Port.Equality
        if rule is Port.Extensive:
            if name.endswith('_split') or name.endswith('_equality') or name == 'splitfrac':
                raise ValueError("Extensive variable '%s' on Port '%s' may not end with '_split' or '_equality'" % (name, self.name))
        self._rules[name] = (rule, kwds)

    def remove(self, name):
        """Remove this member from the port"""
        if name not in self.vars:
            raise ValueError("Cannot remove member '%s' not in Port '%s'" % (name, self.name))
        self.vars.pop(name)
        self._rules.pop(name)

    def rule_for(self, name):
        """Return the rule associated with the given port member"""
        return self._rules[name][0]

    def is_equality(self, name):
        """Return True if the rule for this port member is Port.Equality"""
        return self.rule_for(name) is Port.Equality

    def is_extensive(self, name):
        """Return True if the rule for this port member is Port.Extensive"""
        return self.rule_for(name) is Port.Extensive

    def fix(self):
        """
        Fix all variables in the port at their current values.
        For expressions, fix every variable in the expression.
        """
        for v in self.iter_vars(expr_vars=True, fixed=False):
            v.fix()

    def unfix(self):
        """
        Unfix all variables in the port.
        For expressions, unfix every variable in the expression.
        """
        for v in self.iter_vars(expr_vars=True, fixed=True):
            v.unfix()
    free = unfix

    def iter_vars(self, expr_vars=False, fixed=None, names=False):
        """
        Iterate through every member of the port, going through
        the indices of indexed members.

        Arguments
        ---------
            expr_vars: `bool`
                If True, call `identify_variables` on expression type members
            fixed: `bool`
                Only include variables/expressions with this type of fixed
            names: `bool`
                If True, yield (name, index, var/expr) tuples
        """
        for name, mem in self.vars.items():
            if not mem.is_indexed():
                itr = {None: mem}
            else:
                itr = mem
            for idx, v in itr.items():
                if fixed is not None and v.is_fixed() != fixed:
                    continue
                if expr_vars and v.is_expression_type():
                    for var in identify_variables(v):
                        if fixed is not None and var.is_fixed() != fixed:
                            continue
                        if names:
                            yield (name, idx, var)
                        else:
                            yield var
                elif names:
                    yield (name, idx, v)
                else:
                    yield v

    def set_split_fraction(self, arc, val, fix=True):
        """
        Set the split fraction value to be used for an arc during
        arc expansion when using `Port.Extensive`.
        """
        if arc not in self.dests():
            raise ValueError("Port '%s' is not a source of Arc '%s', cannot set split fraction" % (self.name, arc.name))
        self._splitfracs[arc] = (val, fix)

    def get_split_fraction(self, arc):
        """
        Returns a tuple (val, fix) for the split fraction of this arc that
        was set via `set_split_fraction` if it exists, and otherwise None.
        """
        res = self._splitfracs.get(arc, None)
        if res is None:
            return None
        else:
            return res