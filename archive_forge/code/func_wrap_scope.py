import abc
import functools
from typing import cast, Callable, Set, TypeVar
def wrap_scope(impl: T) -> T:

    def impl_of_abstract(*args, **kwargs):
        return impl(*args, **kwargs)
    functools.update_wrapper(impl_of_abstract, abstract_method)
    return cast(T, impl_of_abstract)