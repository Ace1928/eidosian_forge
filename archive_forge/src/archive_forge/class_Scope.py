from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class Scope(EqualityCheckMixin):
    """
    Attributes
    -----------
    - parent: Scope
        Parent scope

    - localvars: VarMap
        Scope-local variable map

    - loc: Loc
        Start of scope location

    """

    def __init__(self, parent, loc):
        assert parent is None or isinstance(parent, Scope)
        assert isinstance(loc, Loc)
        self.parent = parent
        self.localvars = VarMap()
        self.loc = loc
        self.redefined = defaultdict(int)
        self.var_redefinitions = defaultdict(set)

    def define(self, name, loc):
        """
        Define a variable
        """
        v = Var(scope=self, name=name, loc=loc)
        self.localvars.define(v.name, v)
        return v

    def get(self, name):
        """
        Refer to a variable.  Returns the latest version.
        """
        if name in self.redefined:
            name = '%s.%d' % (name, self.redefined[name])
        return self.get_exact(name)

    def get_exact(self, name):
        """
        Refer to a variable.  The returned variable has the exact
        name (exact variable version).
        """
        try:
            return self.localvars.get(name)
        except NotDefinedError:
            if self.has_parent:
                return self.parent.get(name)
            else:
                raise

    def get_or_define(self, name, loc):
        if name in self.redefined:
            name = '%s.%d' % (name, self.redefined[name])
        if name not in self.localvars:
            return self.define(name, loc)
        else:
            return self.localvars.get(name)

    def redefine(self, name, loc, rename=True):
        """
        Redefine if the name is already defined
        """
        if name not in self.localvars:
            return self.define(name, loc)
        elif not rename:
            return self.localvars.get(name)
        else:
            while True:
                ct = self.redefined[name]
                self.redefined[name] = ct + 1
                newname = '%s.%d' % (name, ct + 1)
                try:
                    res = self.define(newname, loc)
                except RedefinedError:
                    continue
                else:
                    self.var_redefinitions[name].add(newname)
                return res

    def get_versions_of(self, name):
        """
        Gets all known versions of a given name
        """
        vers = set()

        def walk(thename):
            redefs = self.var_redefinitions.get(thename, None)
            if redefs:
                for v in redefs:
                    vers.add(v)
                    walk(v)
        walk(name)
        return vers

    def make_temp(self, loc):
        n = len(self.localvars)
        v = Var(scope=self, name='$%d' % n, loc=loc)
        self.localvars.define(v.name, v)
        return v

    @property
    def has_parent(self):
        return self.parent is not None

    def __repr__(self):
        return 'Scope(has_parent=%r, num_vars=%d, %s)' % (self.has_parent, len(self.localvars), self.loc)