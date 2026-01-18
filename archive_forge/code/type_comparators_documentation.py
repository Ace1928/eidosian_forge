from typing import cast
from ..type import (
Check whether two types overlap in a given schema.

    Provided two composite types, determine if they "overlap". Two composite types
    overlap when the Sets of possible concrete types for each intersect.

    This is often used to determine if a fragment of a given type could possibly be
    visited in a context of another type.

    This function is commutative.
    