from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def add_deprecation_to_docstring(func: Callable, msg: str, *, since: str | None, pending: bool) -> None:
    """Dynamically insert the deprecation message into ``func``'s docstring.

    Args:
        func: The function to modify.
        msg: The full deprecation message.
        since: The version the deprecation started at.
        pending: Is the deprecation still pending?
    """
    if '\n' in msg:
        raise ValueError(f'Deprecation messages cannot contain new lines (`\\n`), but the deprecation for {func.__qualname__} had them. Usually this happens when using `"""` multiline strings; instead, use string concatenation.\n\nThis is a simplification to facilitate deprecation messages being added to the documentation. If you have a compelling reason to need new lines, feel free to improve this function or open a request at https://github.com/Qiskit/qiskit/issues.')
    if since is None:
        version_str = 'unknown'
    else:
        version_str = f'{since}_pending' if pending else since
    indent = ''
    meta_index = None
    if func.__doc__:
        original_lines = func.__doc__.splitlines()
        content_encountered = False
        for i, line in enumerate(original_lines):
            stripped = line.strip()
            if not content_encountered and i != 0 and stripped:
                num_leading_spaces = len(line) - len(line.lstrip())
                indent = ' ' * num_leading_spaces
                content_encountered = True
            if stripped.lower() in _NAPOLEON_META_LINES:
                meta_index = i
                if not content_encountered:
                    raise ValueError(f'''add_deprecation_to_docstring cannot currently handle when a Napoleon metadata line like 'Args' is the very first line of docstring, e.g. `"""Args:`. So, it cannot process {func.__qualname__}. Instead, move the metadata line to the second line, e.g.:\n\n"""\nArgs:''')
                break
    else:
        original_lines = []
    new_lines = [indent, f'{indent}.. deprecated:: {version_str}', f'{indent}  {msg}', indent]
    if meta_index:
        original_lines[meta_index - 1:meta_index - 1] = new_lines
    else:
        original_lines.extend(new_lines)
    func.__doc__ = '\n'.join(original_lines)