import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def wrap_description(text, indentation, wrap_length, force_wrap, strict, rest_sections, style: str='sphinx'):
    """Return line-wrapped description text.

    We only wrap simple descriptions. We leave doctests, multi-paragraph text, and
    bulleted lists alone.

    Parameters
    ----------
    text : str
        The unwrapped description text.
    indentation : str
        The indentation string.
    wrap_length : int
        The line length at which to wrap long lines.
    force_wrap : bool
        Whether to force docformatter to wrap long lines when normally they
        would remain untouched.
    strict : bool
        Whether to strictly follow reST syntax to identify lists.
    rest_sections : str
        A regular expression used to find reST section header adornments.
    style : str
        The name of the docstring style to use when dealing with parameter
        lists (default is sphinx).

    Returns
    -------
    description : str
        The description wrapped at wrap_length characters.
    """
    text = strip_leading_blank_lines(text)
    if '>>>' in text:
        return text
    text = reindent(text, indentation).rstrip()
    if wrap_length <= 0 or (not force_wrap and (is_some_sort_of_code(text) or do_find_directives(text) or is_some_sort_of_list(text, strict, rest_sections, style))):
        return text
    lines = do_split_description(text, indentation, wrap_length, style)
    return indentation + '\n'.join(lines).strip()