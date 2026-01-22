import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _rows_to_latex(rows, max_width):
    row_string = ''
    for r in rows:
        if len(r) <= max_width:
            row_string += _elements_to_latex(r)
        else:
            row_string += _elements_to_latex(r[:max_width // 2])
            row_string += '& \\cdots & '
            row_string += _elements_to_latex(r[-max_width // 2 + 1:])
        row_string += ' \\\\\n '
    return row_string