import abc
import collections
import collections.abc
import operator
import sys
import typing
class _Literal(typing._FinalTypingBase, _root=True):
    """A type that can be used to indicate to type checkers that the
        corresponding value has a value literally equivalent to the
        provided parameter. For example:

            var: Literal[4] = 4

        The type checker understands that 'var' is literally equal to the
        value 4 and no other value.

        Literal[...] cannot be subclassed. There is no runtime checking
        verifying that the parameter is actually a value instead of a type.
        """
    __slots__ = ('__values__',)

    def __init__(self, values=None, **kwds):
        self.__values__ = values

    def __getitem__(self, values):
        cls = type(self)
        if self.__values__ is None:
            if not isinstance(values, tuple):
                values = (values,)
            return cls(values, _root=True)
        raise TypeError(f'{cls.__name__[1:]} cannot be further subscripted')

    def _eval_type(self, globalns, localns):
        return self

    def __repr__(self):
        r = super().__repr__()
        if self.__values__ is not None:
            r += f'[{', '.join(map(typing._type_repr, self.__values__))}]'
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__values__))

    def __eq__(self, other):
        if not isinstance(other, _Literal):
            return NotImplemented
        if self.__values__ is not None:
            return self.__values__ == other.__values__
        return self is other