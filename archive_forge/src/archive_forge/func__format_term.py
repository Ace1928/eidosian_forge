import numbers
from typing import (
from typing_extensions import Self
def _format_term(format_spec: str, vector: TVector, coefficient: Scalar) -> str:
    coefficient_str = _format_coefficient(format_spec, coefficient)
    if not coefficient_str:
        return coefficient_str
    result = f'{coefficient_str}*{vector!s}'
    if result[0] in ['+', '-']:
        return result
    return '+' + result