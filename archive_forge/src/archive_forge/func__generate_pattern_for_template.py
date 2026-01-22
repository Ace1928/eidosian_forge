from __future__ import unicode_literals
from collections import deque
import copy
import functools
import re
def _generate_pattern_for_template(tmpl):
    """Generate a pattern that can validate a path template.

    Args:
        tmpl (str): The path template

    Returns:
        str: A regular expression pattern that can be used to validate an
            expanded path template.
    """
    return _VARIABLE_RE.sub(_replace_variable_with_pattern, tmpl)