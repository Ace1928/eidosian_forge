from collections import namedtuple
from io import StringIO
from keyword import iskeyword
import tokenize
from tokenize import TokenInfo
from typing import List, Optional
Get the token at a given cursor

    Used for introspection.

    Function calls are prioritized, so the token for the callable will be returned
    if the cursor is anywhere inside the call.

    Parameters
    ----------
    cell : str
        A block of Python code
    cursor_pos : int
        The location of the cursor in the block where the token should be found
    