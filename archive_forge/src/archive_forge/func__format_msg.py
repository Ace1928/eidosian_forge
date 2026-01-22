import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def _format_msg(msg: str, *args: Any, no_format: bool=None, _tags: Dict[str, Any]=None, _numbered: Tuple[str, int, int]=None, **kwargs: Any):
    """Formats a message for printing.

    Renders `msg` using the built-in `str.format` and the passed-in
    `*args` and `**kwargs`.

    Args:
        *args (Any): `.format` arguments for `msg`.
        no_format (bool):
            If `no_format` is `True`,
            `.format` will not be called on the message.

            Useful if the output is user-provided or may otherwise
            contain an unexpected formatting string (e.g. "{}").
        _tags (Dict[str, Any]):
            key-value pairs to display at the end of
            the message in square brackets.

            If a tag is set to `True`, it is printed without the value,
            the presence of the tag treated as a "flag".

            E.g. `_format_msg("hello", _tags=dict(from=mom, signed=True))`
                 `hello [from=Mom, signed]`
        _numbered (Tuple[str, int, int]):
            `(brackets, i, n)`

            The `brackets` string is composed of two "bracket" characters,
            `i` is the index, `n` is the total.

            The string `{i}/{n}` surrounded by the "brackets" is
            prepended to the message.

            This is used to number steps in a procedure, with different
            brackets specifying different major tasks.

            E.g. `_format_msg("hello", _numbered=("[]", 0, 5))`
                 `[0/5] hello`

    Returns:
        The formatted message.
    """
    if isinstance(msg, str) or isinstance(msg, ColorfulString):
        tags_str = ''
        if _tags is not None:
            tags_list = []
            for k, v in _tags.items():
                if v is True:
                    tags_list += [k]
                    continue
                if v is False:
                    continue
                tags_list += [k + '=' + v]
            if tags_list:
                tags_str = cf.reset(cf.dimmed(' [{}]'.format(', '.join(tags_list))))
        numbering_str = ''
        if _numbered is not None:
            chars, i, n = _numbered
            numbering_str = cf.dimmed(chars[0] + str(i) + '/' + str(n) + chars[1]) + ' '
        if no_format:
            return numbering_str + msg + tags_str
        return numbering_str + msg.format(*args, **kwargs) + tags_str
    if kwargs:
        raise ValueError('We do not support printing kwargs yet.')
    res = [msg, *args]
    res = [str(x) for x in res]
    return ', '.join(res)