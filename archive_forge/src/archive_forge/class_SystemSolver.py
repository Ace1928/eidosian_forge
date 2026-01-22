import copy
from collections import OrderedDict
from math import log2
import numpy as np
from .. import functions as fn
class SystemSolver(object):
    """
    This abstract class is used to formalize and manage user interaction with a 
    complex system of equations (related to "constraint satisfaction problems").
    It is often the case that devices must be controlled
    through a large number of free variables, and interactions between these 
    variables make the system difficult to manage and conceptualize as a user
    interface. This class does _not_ attempt to numerically solve the system
    of equations. Rather, it provides a framework for subdividing the system
    into manageable pieces and specifying closed-form solutions to these small 
    pieces.
    
    For an example, see the simple Camera class below.
    
    Theory of operation: Conceptualize the system as 1) a set of variables
    whose values may be either user-specified or automatically generated, and 
    2) a set of functions that define *how* each variable should be generated. 
    When a variable is accessed (as an instance attribute), the solver first
    checks to see if it already has a value (either user-supplied, or cached
    from a previous calculation). If it does not, then the solver calls a 
    method on itself (the method must be named `_variableName`) that will
    either return the calculated value (which usually involves acccessing
    other variables in the system), or raise RuntimeError if it is unable to
    calculate the value (usually because the user has not provided sufficient
    input to fully constrain the system). 
    
    Each method that calculates a variable value may include multiple 
    try/except blocks, so that if one method generates a RuntimeError, it may 
    fall back on others. 
    In this way, the system may be solved by recursively searching the tree of 
    possible relationships between variables. This allows the user flexibility
    in deciding which variables are the most important to specify, while 
    avoiding the apparent combinatorial explosion of calculation pathways
    that must be considered by the developer.
    
    Solved values are cached for efficiency, and automatically cleared when 
    a state change invalidates the cache. The rules for this are simple: any
    time a value is set, it invalidates the cache *unless* the previous value
    was None (which indicates that no other variable has yet requested that 
    value). More complex cache management may be defined in subclasses.
    
    
    Subclasses must define:
    
    1) The *defaultState* class attribute: This is a dict containing a 
       description of the variables in the system--their default values,
       data types, and the ways they can be constrained. The format is::

           { name: [value, type, constraint, allowed_constraints], ...}

       Where:
         * *value* is the default value. May be None if it has not been specified
           yet.
         * *type* may be float, int, bool, np.ndarray, ...
         * *constraint* may be None, single value, or (min, max)
              * None indicates that the value is not constrained--it may be
                automatically generated if the value is requested.
         * *allowed_constraints* is a string composed of (n)one, (f)ixed, and (r)ange.
       
       Note: do not put mutable objects inside defaultState!
       
    2) For each variable that may be automatically determined, a method must 
       be defined with the name `_variableName`. This method may either return
       the 
    """
    defaultState = OrderedDict()

    def __init__(self):
        self.__dict__['_vars'] = OrderedDict()
        self.__dict__['_currentGets'] = set()
        self.reset()

    def copy(self):
        sys = type(self)()
        sys.__dict__['_vars'] = copy.deepcopy(self.__dict__['_vars'])
        sys.__dict__['_currentGets'] = copy.deepcopy(self.__dict__['_currentGets'])
        return sys

    def reset(self):
        """
        Reset all variables in the solver to their default state.
        """
        self._currentGets.clear()
        for k in self.defaultState:
            self._vars[k] = self.defaultState[k][:]

    def __getattr__(self, name):
        if name in self._vars:
            return self.get(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        Set the value of a state variable. 
        If None is given for the value, then the constraint will also be set to None.
        If a tuple is given for a scalar variable, then the tuple is used as a range constraint instead of a value.
        Otherwise, the constraint is set to 'fixed'.
        
        """
        if name in self._vars:
            if value is None:
                self.set(name, value, None)
            elif isinstance(value, tuple) and self._vars[name][1] is not np.ndarray:
                self.set(name, None, value)
            else:
                self.set(name, value, 'fixed')
        elif hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(name)

    def get(self, name):
        """
        Return the value for parameter *name*. 
        
        If the value has not been specified, then attempt to compute it from
        other interacting parameters.
        
        If no value can be determined, then raise RuntimeError.
        """
        if name in self._currentGets:
            raise RuntimeError("Cyclic dependency while calculating '%s'." % name)
        self._currentGets.add(name)
        try:
            v = self._vars[name][0]
            if v is None:
                cfunc = getattr(self, '_' + name, None)
                if cfunc is None:
                    v = None
                else:
                    v = cfunc()
                if v is None:
                    raise RuntimeError("Parameter '%s' is not specified." % name)
                v = self.set(name, v)
        finally:
            self._currentGets.remove(name)
        return v

    def set(self, name, value=None, constraint=True):
        """
        Set a variable *name* to *value*. The actual set value is returned (in
        some cases, the value may be cast into another type).
        
        If *value* is None, then the value is left to be determined in the 
        future. At any time, the value may be re-assigned arbitrarily unless
        a constraint is given.
        
        If *constraint* is True (the default), then supplying a value that 
        violates a previously specified constraint will raise an exception.
        
        If *constraint* is 'fixed', then the value is set (if provided) and
        the variable will not be updated automatically in the future.

        If *constraint* is a tuple, then the value is constrained to be within the 
        given (min, max). Either constraint may be None to disable 
        it. In some cases, a constraint cannot be satisfied automatically,
        and the user will be forced to resolve the constraint manually.
        
        If *constraint* is None, then any constraints are removed for the variable.
        """
        var = self._vars[name]
        if constraint is None:
            if 'n' not in var[3]:
                raise TypeError("Empty constraints not allowed for '%s'" % name)
            var[2] = constraint
        elif constraint == 'fixed':
            if 'f' not in var[3]:
                raise TypeError("Fixed constraints not allowed for '%s'" % name)
            var[2] = constraint
        elif isinstance(constraint, tuple):
            if 'r' not in var[3]:
                raise TypeError("Range constraints not allowed for '%s'" % name)
            assert len(constraint) == 2
            var[2] = constraint
        elif constraint is not True:
            raise TypeError("constraint must be None, True, 'fixed', or tuple. (got %s)" % constraint)
        if var[1] is np.ndarray and value is not None:
            value = np.array(value, dtype=float)
        elif var[1] in (int, float, tuple) and value is not None:
            value = var[1](value)
        if constraint is True and (not self.check_constraint(name, value)):
            raise ValueError('Setting %s = %s violates constraint %s' % (name, value, var[2]))
        if var[0] is not None or value is None:
            self.resetUnfixed()
        var[0] = value
        return value

    def check_constraint(self, name, value):
        c = self._vars[name][2]
        if c is None or value is None:
            return True
        if isinstance(c, tuple):
            return (c[0] is None or c[0] <= value) and (c[1] is None or c[1] >= value)
        else:
            return value == c

    def saveState(self):
        """
        Return a serializable description of the solver's current state.
        """
        state = OrderedDict()
        for name, var in self._vars.items():
            state[name] = (var[0], var[2])
        return state

    def restoreState(self, state):
        """
        Restore the state of all values and constraints in the solver.
        """
        self.reset()
        for name, var in state.items():
            self.set(name, var[0], var[1])

    def resetUnfixed(self):
        """
        For any variable that does not have a fixed value, reset
        its value to None.
        """
        for var in self._vars.values():
            if var[2] != 'fixed':
                var[0] = None

    def solve(self):
        for k in self._vars:
            getattr(self, k)

    def checkOverconstraint(self):
        """Check whether the system is overconstrained. If so, return the name of
        the first overconstrained parameter.

        Overconstraints occur when any fixed parameter can be successfully computed by the system.
        (Ideally, all parameters are either fixed by the user or constrained by the
        system, but never both).
        """
        for k, v in self._vars.items():
            if v[2] == 'fixed' and 'n' in v[3]:
                oldval = v[:]
                self.set(k, None, None)
                try:
                    self.get(k)
                    return k
                except RuntimeError:
                    pass
                finally:
                    self._vars[k] = oldval
        return False

    def __repr__(self):
        state = OrderedDict()
        for name, var in self._vars.items():
            if var[2] == 'fixed':
                state[name] = var[0]
        state = ', '.join(['%s=%s' % (n, v) for n, v in state.items()])
        return '<%s %s>' % (self.__class__.__name__, state)