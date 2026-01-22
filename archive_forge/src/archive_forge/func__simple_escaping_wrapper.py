import functools
import re
import string
import sys
import typing as t
def _simple_escaping_wrapper(func: 't.Callable[_P, str]') -> 't.Callable[_P, Markup]':

    @functools.wraps(func)
    def wrapped(self: 'Markup', *args: '_P.args', **kwargs: '_P.kwargs') -> 'Markup':
        arg_list = _escape_argspec(list(args), enumerate(args), self.escape)
        _escape_argspec(kwargs, kwargs.items(), self.escape)
        return self.__class__(func(self, *arg_list, **kwargs))
    return wrapped