from typing import Any, Union, Iterable
from fractions import Fraction
from decimal import Decimal
import numbers
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
Approximate comparator.

        Types implementing this protocol define their own logic for approximate
        comparison with other types.

        Args:
            other: Target object for approximate comparison.
            atol: The minimum absolute tolerance. See np.isclose() documentation
                  for details.

        Returns:
            True if objects are approximately equal, False otherwise. Returns
            NotImplemented when approximate equality is not implemented for
            given types.
        