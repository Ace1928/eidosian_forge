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
class Var(EqualityCheckMixin, AbstractRHS):
    """
    Attributes
    -----------
    - scope: Scope

    - name: str

    - loc: Loc
        Definition location
    """

    def __init__(self, scope, name, loc):
        assert scope is None or isinstance(scope, Scope)
        assert isinstance(name, str)
        assert isinstance(loc, Loc)
        self.scope = scope
        self.name = name
        self.loc = loc

    def __repr__(self):
        return 'Var(%s, %s)' % (self.name, self.loc.short())

    def __str__(self):
        return self.name

    @property
    def is_temp(self):
        return self.name.startswith('$')

    @property
    def unversioned_name(self):
        """The unversioned name of this variable, i.e. SSA renaming removed
        """
        for k, redef_set in self.scope.var_redefinitions.items():
            if self.name in redef_set:
                return k
        return self.name

    @property
    def versioned_names(self):
        """Known versioned names for this variable, i.e. known variable names in
        the scope that have been formed from applying SSA to this variable
        """
        return self.scope.get_versions_of(self.unversioned_name)

    @property
    def all_names(self):
        """All known versioned and unversioned names for this variable
        """
        return self.versioned_names | {self.unversioned_name}

    def __deepcopy__(self, memo):
        out = Var(copy.deepcopy(self.scope, memo), self.name, self.loc)
        memo[id(self)] = out
        return out