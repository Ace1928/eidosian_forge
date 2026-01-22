import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def _formatted_exponent(self, args: 'cirq.CircuitDiagramInfoArgs') -> Optional[str]:
    if protocols.is_parameterized(self.exponent):
        name = str(self.exponent)
        return f'({name})' if _is_exposed_formula(name) else name
    if self.exponent == 0:
        return '0'
    if self.exponent == 1:
        return None
    if self.exponent == -1:
        return '-1'
    if isinstance(self.exponent, float):
        if args.precision is not None:
            approx_frac = Fraction(self.exponent).limit_denominator(16)
            if approx_frac.denominator not in [2, 4, 5, 10]:
                if abs(float(approx_frac) - self.exponent) < 10 ** (-args.precision):
                    return f'({approx_frac})'
            return args.format_real(self.exponent)
        return repr(self.exponent)
    s = str(self.exponent)
    if self.auto_exponent_parens and ('+' in s or ' ' in s or '-' in s[1:]):
        return f'({self.exponent})'
    return s