from typing import FrozenSet, Mapping, Optional, Tuple
import dataclasses
Adds the input path component to the start of the path.

        Useful when constructing the path from inside to out (in case of nested subcircuits),
        recursively.
        