import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _matrix_to_latex(matrix, decimals=10, prefix='', max_size=(8, 8)):
    """Latex representation of a complex numpy array (with maximum dimension 2)

    Args:
        matrix (ndarray): The matrix to be converted to latex, must have dimension 2.
        decimals (int): For numbers not close to integers, the number of decimal places
                         to round to.
        prefix (str): Latex string to be prepended to the latex, intended for labels.
        max_size (list(```int```)): Indexable containing two integers: Maximum width and maximum
                          height of output Latex matrix (including dots characters). If the
                          width and/or height of matrix exceeds the maximum, the centre values
                          will be replaced with dots. Maximum width or height must be greater
                          than 3.

    Returns:
        str: Latex representation of the matrix

    Raises:
        ValueError: If minimum value in max_size < 3
    """
    if min(max_size) < 3:
        raise ValueError('Smallest value in max_size must be greater than or equal to 3')
    out_string = f'\n{prefix}\n'
    out_string += '\\begin{bmatrix}\n'

    def _elements_to_latex(elements):
        el_string = ''
        for el in elements:
            num_string = _num_to_latex(el, decimals=decimals)
            el_string += num_string + ' & '
        el_string = el_string[:-2]
        return el_string

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
    max_width, max_height = max_size
    if matrix.ndim == 1:
        out_string += _rows_to_latex([matrix], max_width)
    elif len(matrix) > max_height:
        out_string += _rows_to_latex(matrix[:max_height // 2], max_width)
        if max_width >= matrix.shape[1]:
            out_string += '\\vdots & ' * matrix.shape[1]
        else:
            pre_vdots = max_width // 2
            post_vdots = max_width // 2 + np.mod(max_width, 2) - 1
            out_string += '\\vdots & ' * pre_vdots
            out_string += '\\ddots & '
            out_string += '\\vdots & ' * post_vdots
        out_string = out_string[:-2] + '\\\\\n '
        out_string += _rows_to_latex(matrix[-max_height // 2 + 1:], max_width)
    else:
        out_string += _rows_to_latex(matrix, max_width)
    out_string += '\\end{bmatrix}\n'
    return out_string