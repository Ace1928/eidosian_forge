import abc
import collections
import collections.abc
import operator
import sys
import typing
class _ExtensionsGenericMeta(GenericMeta):

    def __subclasscheck__(self, subclass):
        """This mimics a more modern GenericMeta.__subclasscheck__() logic
        (that does not have problems with recursion) to work around interactions
        between collections, typing, and typing_extensions on older
        versions of Python, see https://github.com/python/typing/issues/501.
        """
        if self.__origin__ is not None:
            if sys._getframe(1).f_globals['__name__'] not in ['abc', 'functools']:
                raise TypeError('Parameterized generics cannot be used with class or instance checks')
            return False
        if not self.__extra__:
            return super().__subclasscheck__(subclass)
        res = self.__extra__.__subclasshook__(subclass)
        if res is not NotImplemented:
            return res
        if self.__extra__ in subclass.__mro__:
            return True
        for scls in self.__extra__.__subclasses__():
            if isinstance(scls, GenericMeta):
                continue
            if issubclass(subclass, scls):
                return True
        return False