from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    Defines a partial order ('depends on' function) where `these` must occur
    before `those`. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [
        these_before_those_pass_constraint(pass_a, pass_b)
    ]
    ```

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

    def depends_on(a: Callable, b: Callable):
        if unwrap(a) == those and unwrap(b) == these:
            return False
        return True
    return depends_on