import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    a string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(self, load_func: t.Callable[[str], t.Optional[t.Union[str, t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]]]]) -> None:
        self.load_func = load_func

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        rv = self.load_func(template)
        if rv is None:
            raise TemplateNotFound(template)
        if isinstance(rv, str):
            return (rv, None, None)
        return rv