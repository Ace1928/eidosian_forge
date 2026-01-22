from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def any_open_for_or_lambda(self) -> bool:
    """Return True if there is an open for or lambda expression on the line.

        See maybe_increment_for_loop_variable and maybe_increment_lambda_arguments
        for details."""
    return bool(self._for_loop_depths or self._lambda_argument_depths)