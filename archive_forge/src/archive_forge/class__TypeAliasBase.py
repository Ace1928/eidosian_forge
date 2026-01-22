import abc
import collections
import collections.abc
import operator
import sys
import typing
class _TypeAliasBase(typing._FinalTypingBase, metaclass=_TypeAliasMeta, _root=True):
    """Special marker indicating that an assignment should
        be recognized as a proper type alias definition by type
        checkers.

        For example::

            Predicate: TypeAlias = Callable[..., bool]

        It's invalid when used anywhere except as in the example above.
        """
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError('TypeAlias cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError('TypeAlias cannot be used with issubclass().')

    def __repr__(self):
        return 'typing_extensions.TypeAlias'