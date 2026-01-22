from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
class FunctionalCategory(AbstractCCGCategory):
    """
    Class that represents a function application category.
    Consists of argument and result categories, together with
    an application direction.
    """

    def __init__(self, res, arg, dir):
        self._res = res
        self._arg = arg
        self._dir = dir
        self._comparison_key = (arg, dir, res)

    def is_primitive(self):
        return False

    def is_function(self):
        return True

    def is_var(self):
        return False

    def substitute(self, subs):
        sub_res = self._res.substitute(subs)
        sub_dir = self._dir.substitute(subs)
        sub_arg = self._arg.substitute(subs)
        return FunctionalCategory(sub_res, sub_arg, self._dir)

    def can_unify(self, other):
        if other.is_var():
            return [(other, self)]
        if other.is_function():
            sa = self._res.can_unify(other.res())
            sd = self._dir.can_unify(other.dir())
            if sa is not None and sd is not None:
                sb = self._arg.substitute(sa).can_unify(other.arg().substitute(sa))
                if sb is not None:
                    return sa + sb
        return None

    def arg(self):
        return self._arg

    def res(self):
        return self._res

    def dir(self):
        return self._dir

    def __str__(self):
        return f'({self._res}{self._dir}{self._arg})'