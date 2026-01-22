from __future__ import annotations
import re
from typing import (
def _variable(self, name: str, vars_set: set[str]) -> None:
    """Track that `name` is used as a variable.

        Adds the name to `vars_set`, a set of variable names.

        Raises an syntax error if `name` is not a valid name.

        """
    if not re.match('[_a-zA-Z][_a-zA-Z0-9]*$', name):
        self._syntax_error('Not a valid name', name)
    vars_set.add(name)