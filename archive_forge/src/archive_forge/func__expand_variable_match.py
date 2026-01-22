from __future__ import unicode_literals
from collections import deque
import copy
import functools
import re
def _expand_variable_match(positional_vars, named_vars, match):
    """Expand a matched variable with its value.

    Args:
        positional_vars (list): A list of positional variables. This list will
            be modified.
        named_vars (dict): A dictionary of named variables.
        match (re.Match): A regular expression match.

    Returns:
        str: The expanded variable to replace the match.

    Raises:
        ValueError: If a positional or named variable is required by the
            template but not specified or if an unexpected template expression
            is encountered.
    """
    positional = match.group('positional')
    name = match.group('name')
    if name is not None:
        try:
            return str(named_vars[name])
        except KeyError:
            raise ValueError("Named variable '{}' not specified and needed by template `{}` at position {}".format(name, match.string, match.start()))
    elif positional is not None:
        try:
            return str(positional_vars.pop(0))
        except IndexError:
            raise ValueError('Positional variable not specified and needed by template `{}` at position {}'.format(match.string, match.start()))
    else:
        raise ValueError('Unknown template expression {}'.format(match.group(0)))