import abc
import functools
from typing import cast, Callable, Set, TypeVar
class ABCMetaImplementAnyOneOf(abc.ABCMeta):
    """A metaclass extending `abc.ABCMeta` for defining flexible abstract base classes

    This metadata allows the declaration of an abstract base classe (ABC)
    with more flexibility in which methods must be overridden.

    Use this metaclass in the same way as `abc.ABCMeta` to create an ABC.

    In addition to the decorators in the` abc` module, the decorator
    `@alternative(...)` may be used.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        implemented_by = {}

        def has_some_implementation(name: str) -> bool:
            if name in implemented_by:
                return True
            try:
                value = getattr(cls, name)
            except AttributeError:
                raise TypeError(f"A method named '{name}' was listed as a possible implementation alternative but it does not exist in the definition of {cls!r}.")
            if getattr(value, '__isabstractmethod__', False):
                return False
            if hasattr(value, '_abstract_alternatives_'):
                return False
            return True

        def find_next_implementations(all_names: Set[str]) -> bool:
            next_implemented_by = {}
            for name in all_names:
                if has_some_implementation(name):
                    continue
                value = getattr(cls, name, None)
                if not hasattr(value, '_abstract_alternatives_'):
                    continue
                for alt_name, impl in getattr(value, '_abstract_alternatives_'):
                    if has_some_implementation(alt_name):
                        next_implemented_by[name] = impl
                        break
            implemented_by.update(next_implemented_by)
            return bool(next_implemented_by)
        all_names = set((alt_name for alt_name in namespace.keys() if hasattr(cls, alt_name)))
        for base in bases:
            all_names.update(getattr(base, '__abstractmethods__', set()))
            all_names.update((alt_name for alt_name, _ in getattr(base, '_implemented_by_', {}).items()))
        while find_next_implementations(all_names):
            pass
        abstracts = frozenset((name for name in all_names if not has_some_implementation(name)))
        for name, default_impl in implemented_by.items():
            abstract_method = getattr(cls, name)

            def wrap_scope(impl: T) -> T:

                def impl_of_abstract(*args, **kwargs):
                    return impl(*args, **kwargs)
                functools.update_wrapper(impl_of_abstract, abstract_method)
                return cast(T, impl_of_abstract)
            impl_of_abstract = wrap_scope(default_impl)
            impl_of_abstract._abstract_alternatives_ = abstract_method._abstract_alternatives_
            setattr(cls, name, impl_of_abstract)
        cls.__abstractmethods__ |= abstracts
        cls._implemented_by_ = implemented_by
        return cls