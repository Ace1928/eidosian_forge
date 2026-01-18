from typing import Optional
from ._util import to_string

    Internal class used to parse results from the `SUGGET` command.
    This needs to consume either 1, 2, or 3 values at a time from
    the return value depending on what objects were requested
    