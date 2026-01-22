import abc
import functools
from typing import cast, Callable, Set, TypeVar
def alternative(*, requires: str, implementation: T) -> Callable[[T], T]:
    """A decorator indicating an abstract method with an alternative default implementation.

    This decorator may be used multiple times on the same function to specify
    multiple alternatives.  If multiple alternatives are available, the
    outermost (lowest line number) alternative is used.

    Usage:
        class Parent(metaclass=ABCMetaImplementAnyOneOf):
            def _default_do_a_using_b(self, ...):
                ...
            def _default_do_a_using_c(self, ...):
                ...

            # Abstract method with alternatives
            @alternative(requires='do_b', implementation=_default_do_a_using_b)
            @alternative(requires='do_c', implementation=_default_do_a_using_c)
            def do_a(self, ...):
                '''Method docstring.'''

            # Abstract or concrete methods `do_b` and `do_c`:
            ...

        class Child(Parent):
            def do_b(self):
                ...

        child = Child()
        child.do_a(...)

    Arguments:
        requires: The name of another abstract method in the same class that
            `implementation` needs to be implemented.
        implementation: A function that uses the method named by requires to
            implement the default behavior of the wrapped abstract method.  This
            function must have the same signature as the decorated function.
    """

    def decorator(func: T) -> T:
        alternatives = getattr(func, '_abstract_alternatives_', [])
        alternatives.insert(0, (requires, implementation))
        setattr(func, '_abstract_alternatives_', alternatives)
        return func
    return decorator